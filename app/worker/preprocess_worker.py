import asyncio
import json
import os
from dataclasses import asdict, is_dataclass
from io import BytesIO
from typing import Any, Dict, Tuple
from uuid import uuid4

import asyncpg
from minio import Minio
from minio.error import S3Error

from app.services import preprocessing
from app.services import minhash


MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

MINIO_SECURE = os.getenv(
    "MINIO_SECURE",
    "false",
).lower() == "true"

MINHASH_THRESHOLD = float(
    os.getenv("MINHASH_THRESHOLD", "0.05")
)

POSTGRES_DSN = os.getenv("POSTGRES_DSN")
SUBJECT_ID = os.getenv("SUBJECT_ID")


def create_minio_client() -> Minio:
    if not MINIO_ACCESS_KEY:
        raise ValueError("Missing MINIO_ACCESS_KEY")

    if not MINIO_SECRET_KEY:
        raise ValueError("Missing MINIO_SECRET_KEY")

    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def parse_minio_path(path: str) -> Tuple[str, str]:
    if not path.startswith("minio://"):
        raise ValueError(f"Invalid MinIO path: {path}")

    raw = path.replace("minio://", "", 1)

    if "/" not in raw:
        raise ValueError(f"Invalid MinIO path: {path}")

    bucket, object_name = raw.split("/", 1)

    if not bucket:
        raise ValueError(f"Invalid MinIO bucket in path: {path}")

    if not object_name:
        raise ValueError(f"Invalid MinIO object name in path: {path}")

    return bucket, object_name


async def load_pdf_bytes_from_minio(path: str) -> bytes:
    bucket, object_name = parse_minio_path(path)

    client = create_minio_client()

    try:
        response = client.get_object(bucket, object_name)

        try:
            return response.read()

        finally:
            response.close()
            response.release_conn()

    except S3Error as e:
        raise RuntimeError(
            f"Cannot load PDF from MinIO: {path}, error={e}"
        ) from e


async def save_json_to_minio(
    path: str,
    data: Dict[str, Any],
):
    bucket, object_name = parse_minio_path(path)

    client = create_minio_client()

    payload = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    try:
        found = client.bucket_exists(bucket)

        if not found:
            client.make_bucket(bucket)

        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )

    except S3Error as e:
        raise RuntimeError(
            f"Cannot save JSON to MinIO: {path}, error={e}"
        ) from e


def convert_sentences(sentences: list[Any]) -> list[dict]:
    results: list[dict] = []

    for index, sentence in enumerate(sentences):
        if is_dataclass(sentence):
            item = asdict(sentence)

        elif isinstance(sentence, dict):
            item = dict(sentence)

        else:
            item = {
                "page_number": getattr(sentence, "page_number", 0),
                "sentence_index": getattr(sentence, "sentence_index", index),
                "sentence_index_page": getattr(
                    sentence,
                    "sentence_index_page",
                    index,
                ),
                "sentence_text": getattr(sentence, "sentence_text", ""),
                "bbox_x0": getattr(sentence, "bbox_x0", 0.0),
                "bbox_y0": getattr(sentence, "bbox_y0", 0.0),
                "bbox_x1": getattr(sentence, "bbox_x1", 0.0),
                "bbox_y1": getattr(sentence, "bbox_y1", 0.0),
            }

        if not item.get("sentence_text"):
            raise ValueError(
                f"Missing sentence_text at index={index}: {item}"
            )

        results.append(item)

    return results


def normalize_candidates(candidates: list[dict]) -> list[dict]:
    """
    Đảm bảo candidates lưu xuống MinIO có đủ field
    cho checker stage 2 sử dụng.
    """

    normalized: list[dict] = []

    for candidate in candidates:
        item = dict(candidate)

        # Nếu service minhash chưa trả field id,
        # tạm dùng document_id để fallback.
        # Tốt nhất vẫn nên sửa service minhash để trả id rõ ràng.
        if "id" not in item:
            document_id = item.get("document_id")

            if isinstance(document_id, int):
                item["id"] = document_id

            elif isinstance(document_id, str) and document_id.isdigit():
                item["id"] = int(document_id)

            else:
                item["id"] = 0

        item.setdefault("document_id", str(item.get("id", "")))
        item.setdefault("file_name", "")
        item.setdefault("subject_id", SUBJECT_ID or "")
        item.setdefault("jaccard_similarity", 0.0)
        item.setdefault("group_id", None)

        normalized.append(item)

    return normalized


async def find_candidates(
    minhash_values: list[int],
    subject_id: str,
) -> list[dict]:
    if not POSTGRES_DSN:
        raise ValueError("Missing POSTGRES_DSN")

    conn = await asyncpg.connect(POSTGRES_DSN)

    try:
        candidates = await minhash.find_candidates_by_minhash(
            conn=conn,
            minhash=minhash_values,
            subject_id=subject_id,
            threshold=MINHASH_THRESHOLD,
        )

        return normalize_candidates(candidates)

    finally:
        await conn.close()


async def process_job(
    input_pdf_path: str,
    output_json_path: str,
    subject_id: str,
):
    print("=== Start preprocess worker ===", flush=True)

    print(f"INPUT_PDF={input_pdf_path}", flush=True)
    print(f"OUTPUT_JSON={output_json_path}", flush=True)
    print(f"SUBJECT_ID={subject_id}", flush=True)

    pdf_bytes = await load_pdf_bytes_from_minio(input_pdf_path)

    if len(pdf_bytes) == 0:
        raise ValueError("PDF file is empty")

    print(f"Loaded PDF bytes: {len(pdf_bytes)}", flush=True)

    print("Extracting and preprocessing text...", flush=True)

    full_text, sentences = preprocessing.extract_and_preprocess(
        BytesIO(pdf_bytes)
    )

    if not full_text:
        raise ValueError("Cannot extract full_text")

    if not sentences:
        raise ValueError("Cannot extract any sentence")

    print(f"Extracted sentences: {len(sentences)}", flush=True)

    print("Computing MinHash in preprocess stage...", flush=True)

    minhash_values = minhash.compute_minhash(full_text)

    print("Finding candidates by MinHash in preprocess stage...", flush=True)

    candidates = await find_candidates(
        minhash_values=minhash_values,
        subject_id=subject_id,
    )

    print(f"Candidates found: {len(candidates)}", flush=True)

    result = {
        "full_text": full_text,
        "sentences": convert_sentences(sentences),
        "sentence_count": len(sentences),

        # Đây là fingerprint MinHash của tài liệu upload
        "minhash": minhash_values,

        # Đây mới là danh sách candidate sau lọc MinHash
        "candidates": candidates,
        "candidate_count": len(candidates),

        "subject_id": subject_id,
        "minhash_threshold": MINHASH_THRESHOLD,
    }

    await save_json_to_minio(
        output_json_path,
        result,
    )

    print("=== Preprocess completed ===", flush=True)


async def async_main():
    input_pdf_path = os.getenv("INPUT_PDF_PATH")
    output_json_path = os.getenv("OUTPUT_JSON_PATH")
    subject_id = os.getenv("SUBJECT_ID") or SUBJECT_ID

    if not input_pdf_path:
        raise ValueError("Missing INPUT_PDF_PATH")

    if not subject_id:
        raise ValueError("Missing SUBJECT_ID")

    if not output_json_path:
        output_json_path = f"minio://preprocessed/{uuid4()}.json"

        print(
            f"Generated OUTPUT_JSON_PATH={output_json_path}",
            flush=True,
        )

    await process_job(
        input_pdf_path=input_pdf_path,
        output_json_path=output_json_path,
        subject_id=subject_id,
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()