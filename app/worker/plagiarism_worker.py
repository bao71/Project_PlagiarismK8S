import asyncio
import json
import os
from io import BytesIO
from typing import Any, Dict, Tuple
from uuid import uuid4

from minio import Minio
from minio.error import S3Error

from app.services import embedding
from app.services.checker import find_candidate
from app.services.preprocessing import SentenceRecord


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


async def load_json_from_minio(path: str) -> Dict[str, Any]:
    bucket, object_name = parse_minio_path(path)

    client = create_minio_client()

    try:
        response = client.get_object(bucket, object_name)

        try:
            content = response.read().decode("utf-8")
            return json.loads(content)

        finally:
            response.close()
            response.release_conn()

    except S3Error as e:
        raise RuntimeError(
            f"Cannot load JSON from MinIO: {path}, error={e}"
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


def convert_check_response_to_dict(
    check_response: Any,
) -> Dict[str, Any]:
    if hasattr(check_response, "model_dump"):
        return check_response.model_dump()

    if hasattr(check_response, "dict"):
        return check_response.dict()

    if isinstance(check_response, dict):
        return check_response

    raise TypeError(
        f"Unsupported response type: {type(check_response)}"
    )


def convert_sentences_to_records(
    sentences: list[Any],
) -> list[SentenceRecord]:
    sentence_records: list[SentenceRecord] = []

    for index, sentence in enumerate(sentences):
        if isinstance(sentence, SentenceRecord):
            sentence_records.append(sentence)
            continue

        if not isinstance(sentence, dict):
            raise TypeError(
                f"Invalid sentence type at index={index}: "
                f"{type(sentence)}"
            )

        sentence_text = sentence.get("sentence_text")

        if not sentence_text:
            raise ValueError(
                f"Missing sentence_text at index={index}: {sentence}"
            )

        sentence_index = int(
            sentence.get("sentence_index", index)
        )

        normalized = {
            "page_number": int(sentence.get("page_number", 0)),
            "sentence_index": sentence_index,
            "sentence_index_page": int(
                sentence.get("sentence_index_page", sentence_index)
            ),
            "sentence_text": sentence_text,
            "bbox_x0": float(sentence.get("bbox_x0", 0.0)),
            "bbox_y0": float(sentence.get("bbox_y0", 0.0)),
            "bbox_x1": float(sentence.get("bbox_x1", 0.0)),
            "bbox_y1": float(sentence.get("bbox_y1", 0.0)),
        }

        try:
            sentence_records.append(SentenceRecord(**normalized))

        except Exception as e:
            raise ValueError(
                f"Cannot convert sentence at index={index} "
                f"to SentenceRecord. "
                f"sentence={sentence}, "
                f"normalized={normalized}, "
                f"error={e}"
            ) from e

    return sentence_records


def get_candidates_from_preprocess_result(
    preprocess_result: Dict[str, Any],
) -> list[dict]:
    candidates = preprocess_result.get("candidates")

    if candidates is None:
        raise ValueError(
            "Missing candidates in preprocess result. "
            "Stage 1 must run find_candidates_by_minhash() "
            "and save candidates into preprocess result."
        )

    if not isinstance(candidates, list):
        raise TypeError(
            f"Invalid candidates type: {type(candidates)}"
        )

    return candidates


async def process_job(
    subject_id: str,
    preprocess_result_path: str,
    result_output_path: str,
):
    print("=== Start plagiarism worker ===", flush=True)

    print(f"SUBJECT_ID={subject_id}", flush=True)
    print(f"PREPROCESS_RESULT={preprocess_result_path}", flush=True)
    print(f"RESULT_OUTPUT={result_output_path}", flush=True)

    preprocess_result = await load_json_from_minio(
        preprocess_result_path
    )

    raw_sentences = preprocess_result.get("sentences", [])

    print(f"Loaded raw sentences: {len(raw_sentences)}", flush=True)

    if not raw_sentences:
        raise ValueError("No sentences found")

    sentence_records = convert_sentences_to_records(raw_sentences)

    print(
        f"Converted sentence records: {len(sentence_records)}",
        flush=True,
    )

    candidates = get_candidates_from_preprocess_result(
        preprocess_result
    )

    print(
        f"Candidates loaded from preprocess result: {len(candidates)}",
        flush=True,
    )

    if not candidates:
        result = {
            "total_sentences": len(sentence_records),
            "plagiarized_sentences": 0,
            "plagiarism_ratio": 0.0,
            "is_plagiarized": False,
            "sentence_labels": [0] * len(sentence_records),
            "references": [],
        }

        await save_json_to_minio(
            result_output_path,
            result,
        )

        print("=== No plagiarism candidates ===", flush=True)
        return

    sentence_texts = [
        sentence.sentence_text
        for sentence in sentence_records
    ]

    print("Generating embeddings...", flush=True)

    query_embeddings = embedding.embed_sentences(sentence_texts)

    best_candidate = max(
        candidates,
        key=lambda c: c["jaccard_similarity"],
    )

    print(
        f"Best candidate: "
        f"document_id={best_candidate['document_id']}, "
        f"file_name={best_candidate['file_name']}, "
        f"jaccard={best_candidate['jaccard_similarity']}",
        flush=True,
    )
    print("Running plagiarism check with best MinHash candidate only...", flush=True)

    check_response = find_candidate(
        query_sentences=sentence_records,
        query_embeddings=query_embeddings,
        candidates=candidates,
    )

    result = convert_check_response_to_dict(check_response)

    await save_json_to_minio(
        result_output_path,
        result,
    )

    print("=== Plagiarism job completed ===", flush=True)


async def async_main():
    subject_id = os.getenv("SUBJECT_ID")
    preprocess_result_path = os.getenv("PREPROCESS_RESULT_PATH")
    result_output_path = os.getenv("RESULT_OUTPUT_PATH")

    if not subject_id:
        raise ValueError("Missing SUBJECT_ID")

    if not preprocess_result_path:
        raise ValueError("Missing PREPROCESS_RESULT_PATH")

    if not result_output_path:
        result_output_path = f"minio://results/{uuid4()}.json"

        print(
            f"Generated RESULT_OUTPUT_PATH={result_output_path}",
            flush=True,
        )

    await process_job(
        subject_id=subject_id,
        preprocess_result_path=preprocess_result_path,
        result_output_path=result_output_path,
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()