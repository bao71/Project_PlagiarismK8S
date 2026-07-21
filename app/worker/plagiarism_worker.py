import asyncio
import json
import os
import time
from io import BytesIO
from typing import Any

import asyncpg
from minio import Minio
from minio.error import S3Error

from app.services import embedding
from app.services import minhash
from app.services import preprocessing
from app.services.checker import run_plagiarism_check
from app.services.preprocessing import SentenceRecord


MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

MINHASH_POSTGRES_DSN = os.getenv("MINHASH_POSTGRES_DSN")
RESULT_POSTGRES_DSN = os.getenv("RESULT_POSTGRES_DSN")

MINHASH_THRESHOLD = float(os.getenv("MINHASH_THRESHOLD", "0.05"))


async def save_result_to_postgres(
    check_name: str,
    file_id: str,
    topic_id: int,
    subject_id: str,
    file_name: str,
    input_pdf_path: str,
    result: dict,
    candidate_count: int,
) -> None:
    connection = await asyncpg.connect(RESULT_POSTGRES_DSN)

    try:
        command = await connection.execute(
            """
            INSERT INTO public.plagiarism_checks (
                check_name,
                file_id,
                topic_id,
                subject_id,
                file_name,
                input_pdf_path,
                status,
                result,
                candidate_count,
                total_sentences,
                plagiarized_sentences,
                plagiarism_ratio,
                is_plagiarized,
                error_message,
                created_at,
                started_at,
                completed_at,
                updated_at
            )
            VALUES (
                $1::varchar,
                $2::varchar,
                $3::bigint,
                $4::varchar,
                $5::text,
                $6::text,
                'completed',
                $7::jsonb,
                $8::integer,
                $9::integer,
                $10::integer,
                $11::double precision,
                $12::boolean,
                NULL,
                NOW(),
                NOW(),
                NOW(),
                NOW()
            )
            """,
            check_name,
            file_id,
            topic_id,
            subject_id,
            file_name,
            input_pdf_path,
            json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            ),
            candidate_count,
            int(result.get("total_sentences", 0)),
            int(result.get("plagiarized_sentences", 0)),
            float(result.get("plagiarism_ratio", 0.0)),
            bool(result.get("is_plagiarized", False)),
        )

        print(
            f"Saved final result to PostgreSQL: "
            f"check_name={check_name}, command={command}",
            flush=True,
        )
    finally:
        await connection.close()
async def save_failed_result_to_postgres(
    check_name: str,
    file_id: str,
    topic_id: int,
    subject_id: str,
    file_name: str,
    input_pdf_path: str,
    error_message: str,
) -> None:
    connection = await asyncpg.connect(RESULT_POSTGRES_DSN)

    try:
        command = await connection.execute(
            """
            INSERT INTO public.plagiarism_checks (
                check_name,
                file_id,
                topic_id,
                subject_id,
                file_name,
                input_pdf_path,
                status,
                result,
                candidate_count,
                total_sentences,
                plagiarized_sentences,
                plagiarism_ratio,
                is_plagiarized,
                error_message,
                created_at,
                started_at,
                completed_at,
                updated_at
            )
            VALUES (
                $1::varchar,
                $2::varchar,
                $3::bigint,
                $4::varchar,
                $5::text,
                $6::text,
                'failed',
                NULL,
                0,
                0,
                0,
                0.0,
                FALSE,
                $7::text,
                NOW(),
                NOW(),
                NOW(),
                NOW()
            )
            """,
            check_name,
            file_id,
            topic_id,
            subject_id,
            file_name,
            input_pdf_path,
            error_message,
        )

        print(
            f"Saved failed result to PostgreSQL: "
            f"check_name={check_name}, command={command}",
            flush=True,
        )
    finally:
        await connection.close()

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


def parse_minio_path(path: str) -> tuple[str, str]:
    if not path.startswith("minio://"):
        raise ValueError(f"Invalid MinIO path: {path}")

    raw = path.removeprefix("minio://")

    if "/" not in raw:
        raise ValueError(f"Invalid MinIO path: {path}")

    bucket, object_name = raw.split("/", 1)
    return bucket, object_name


async def load_pdf_bytes_from_minio(path: str) -> bytes:
    bucket, object_name = parse_minio_path(path)

    def _load() -> bytes:
        client = create_minio_client()

        try:
            response = client.get_object(bucket, object_name)

            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        except S3Error as exc:
            raise RuntimeError(
                f"Cannot load PDF from MinIO: "
                f"path={path}, error={exc}"
            ) from exc

    return await asyncio.to_thread(_load)


def sentence_record_to_dict(sentence: Any) -> dict:
    if hasattr(sentence, "model_dump"):
        return sentence.model_dump()

    if hasattr(sentence, "dict"):
        return sentence.dict()

    if isinstance(sentence, dict):
        return sentence

    return {
        "page_number": sentence.page_number,
        "sentence_index": sentence.sentence_index,
        "sentence_index_page": sentence.sentence_index_page,
        "sentence_text": sentence.sentence_text,
        "bbox_x0": sentence.bbox_x0,
        "bbox_y0": sentence.bbox_y0,
        "bbox_x1": sentence.bbox_x1,
        "bbox_y1": sentence.bbox_y1,
    }


def normalize_sentence_records(
    sentence_records: list[Any],
) -> list[SentenceRecord]:
    sentence_items = [
        sentence_record_to_dict(sentence)
        for sentence in sentence_records
    ]

    sentence_items.sort(
        key=lambda item: int(item.get("sentence_index", 0))
    )

    normalized_records: list[SentenceRecord] = []

    for position, sentence in enumerate(sentence_items):
        sentence_index = int(
            sentence.get("sentence_index", position)
        )

        if sentence_index != position:
            raise ValueError(
                "Sentence index mismatch: "
                f"list_position={position}, "
                f"sentence_index={sentence_index}"
            )

        sentence_text = sentence.get("sentence_text")

        if not sentence_text:
            raise ValueError(
                "Missing sentence_text at "
                f"sentence_index={sentence_index}"
            )

        normalized_records.append(
            SentenceRecord(
                page_number=int(
                    sentence.get("page_number", 0)
                ),
                sentence_index=sentence_index,
                sentence_index_page=int(
                    sentence.get(
                        "sentence_index_page",
                        sentence_index,
                    )
                ),
                sentence_text=str(sentence_text),
                bbox_x0=float(sentence.get("bbox_x0", 0.0)),
                bbox_y0=float(sentence.get("bbox_y0", 0.0)),
                bbox_x1=float(sentence.get("bbox_x1", 0.0)),
                bbox_y1=float(sentence.get("bbox_y1", 0.0)),
            )
        )

    return normalized_records


def validate_embeddings_alignment(
    sentence_records: list[SentenceRecord],
    query_embeddings: list[list[float]],
) -> None:
    if len(sentence_records) != len(query_embeddings):
        raise ValueError(
            "Embedding count mismatch: "
            f"sentences={len(sentence_records)}, "
            f"embeddings={len(query_embeddings)}"
        )

    for position, vector in enumerate(query_embeddings):
        if len(vector) != 768:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"sentence_index={position}, "
                f"dim={len(vector)}, expected=768"
            )


def convert_check_response_to_dict(
    check_response: Any,
) -> dict:
    if hasattr(check_response, "model_dump"):
        return check_response.model_dump()

    if hasattr(check_response, "dict"):
        return check_response.dict()

    if isinstance(check_response, dict):
        return check_response

    raise TypeError(
        f"Unsupported response type: {type(check_response)}"
    )


async def find_candidates(
    full_text: str,
    subject_id: str,
) -> list[dict]:
    if not MINHASH_POSTGRES_DSN:
        raise ValueError("Missing MINHASH_POSTGRES_DSN")

    minhash_values = await asyncio.to_thread(
        minhash.compute_minhash,
        full_text,
    )

    connection = await asyncpg.connect(MINHASH_POSTGRES_DSN)

    try:
        candidates = await minhash.find_candidates_by_minhash(
            conn=connection,
            minhash=minhash_values,
            subject_id=subject_id,
            threshold=MINHASH_THRESHOLD,
        )
    finally:
        await connection.close()

    return [
        dict(candidate)
        if not isinstance(candidate, dict)
        else candidate
        for candidate in candidates
    ]


async def run_worker() -> None:
    worker_start = time.perf_counter()

    check_name = os.getenv("CHECK_NAME")
    file_id = os.getenv("FILE_ID")
    topic_id_raw = os.getenv("TOPIC_ID")
    subject_id = os.getenv("SUBJECT_ID")
    file_name = os.getenv("FILE_NAME")
    input_pdf_path = os.getenv("INPUT_PDF_PATH")

    if not check_name:
        raise ValueError("Missing CHECK_NAME")

    if not subject_id:
        raise ValueError("Missing SUBJECT_ID")

    if not input_pdf_path:
        raise ValueError("Missing INPUT_PDF_PATH")

    if not MINHASH_POSTGRES_DSN:
        raise ValueError("Missing MINHASH_POSTGRES_DSN")

    if not RESULT_POSTGRES_DSN:
        raise ValueError("Missing RESULT_POSTGRES_DSN")
    if not file_id:
        raise ValueError("Missing FILE_ID")

    if topic_id_raw is None:
        raise ValueError("Missing TOPIC_ID")

    try:
        topic_id = int(topic_id_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid TOPIC_ID: {topic_id_raw}"
        ) from exc

    if not file_name:
        raise ValueError("Missing FILE_NAME")

    print("=== Plagiarism worker starting ===", flush=True)
    print(f"CHECK_NAME={check_name}", flush=True)
    print(f"FILE_ID={file_id}", flush=True)
    print(f"TOPIC_ID={topic_id}", flush=True)
    print(f"SUBJECT_ID={subject_id}", flush=True)
    print(f"FILE_NAME={file_name}", flush=True)
    print(f"INPUT_PDF_PATH={input_pdf_path}", flush=True)

    

    stage_start = time.perf_counter()

    pdf_bytes = await load_pdf_bytes_from_minio(input_pdf_path)

    if not pdf_bytes:
        raise ValueError("Input PDF is empty")

    load_seconds = round(
        time.perf_counter() - stage_start,
        4,
    )

    print(
        f"Loaded input file: bytes={len(pdf_bytes)}, "
        f"seconds={load_seconds}",
        flush=True,
    )

    preprocess_start = time.perf_counter()

    full_text, raw_sentence_records = await asyncio.to_thread(
        preprocessing.extract_and_preprocess,
        BytesIO(pdf_bytes),
    )

    if not full_text:
        raise ValueError("Cannot extract full_text")

    if not raw_sentence_records:
        raise ValueError("Cannot extract any sentence")

    sentence_records = normalize_sentence_records(
        raw_sentence_records
    )

    preprocess_seconds = round(
        time.perf_counter() - preprocess_start,
        4,
    )

    print(
        f"Preprocessed document: "
        f"sentences={len(sentence_records)}, "
        f"seconds={preprocess_seconds}",
        flush=True,
    )

    embedding_start = time.perf_counter()

    sentence_texts = [
        sentence.sentence_text
        for sentence in sentence_records
    ]

    query_embeddings = await asyncio.to_thread(
        embedding.embed_sentences,
        sentence_texts,
    )

    validate_embeddings_alignment(
        sentence_records=sentence_records,
        query_embeddings=query_embeddings,
    )

    embedding_seconds = round(
        time.perf_counter() - embedding_start,
        4,
    )

    print(
        f"Generated query embeddings: "
        f"count={len(query_embeddings)}, "
        f"seconds={embedding_seconds}",
        flush=True,
    )

    candidate_start = time.perf_counter()

    candidates = await find_candidates(
        full_text=full_text,
        subject_id=subject_id,
    )

    candidate_seconds = round(
        time.perf_counter() - candidate_start,
        4,
    )

    print(
        f"Candidates found: "
        f"count={len(candidates)}, "
        f"seconds={candidate_seconds}",
        flush=True,
    )

    check_start = time.perf_counter()

    if not candidates:
        result = {
            "total_sentences": len(sentence_records),
            "plagiarized_sentences": 0,
            "plagiarism_ratio": 0.0,
            "is_plagiarized": False,
            "sentence_labels": [0] * len(sentence_records),
            "references": [],
        }
    else:
        check_response = await asyncio.to_thread(
            run_plagiarism_check,
            query_sentences=sentence_records,
            query_embeddings=query_embeddings,
            candidates=candidates,
        )

        result = convert_check_response_to_dict(
            check_response
        )

    plagiarism_check_seconds = round(
        time.perf_counter() - check_start,
        4,
    )

    total_worker_seconds = round(
        time.perf_counter() - worker_start,
        4,
    )

    result["check_name"] = check_name
    result["subject_id"] = subject_id
    result["candidate_count"] = len(candidates)
    result["load_input_seconds"] = load_seconds
    result["preprocess_seconds"] = preprocess_seconds
    result["embedding_seconds"] = embedding_seconds
    result["candidate_filter_seconds"] = candidate_seconds
    result[
        "plagiarism_check_seconds"
    ] = plagiarism_check_seconds
    result["total_worker_seconds"] = total_worker_seconds

    await save_result_to_postgres(
        check_name=check_name,
        file_id=file_id,
        topic_id=topic_id,
        subject_id=subject_id,
        file_name=file_name,
        input_pdf_path=input_pdf_path,
        result=result,
        candidate_count=len(candidates),
    )

    

    print(
        "PLAGIARISM_CHECK_SECONDS="
        f"{plagiarism_check_seconds}",
        flush=True,
    )
    print(
        "TOTAL_WORKER_SECONDS="
        f"{total_worker_seconds}",
        flush=True,
    )
    print(
        "=== Plagiarism worker completed ===",
        flush=True,
    )

async def main() -> None:
    check_name = os.getenv("CHECK_NAME")
    file_id = os.getenv("FILE_ID")
    topic_id_raw = os.getenv("TOPIC_ID")
    subject_id = os.getenv("SUBJECT_ID")
    file_name = os.getenv("FILE_NAME")
    input_pdf_path = os.getenv("INPUT_PDF_PATH")

    try:
        await run_worker()

    except Exception as exc:
        required_metadata_available = all(
            [
                check_name,
                file_id,
                topic_id_raw is not None,
                subject_id,
                file_name,
                input_pdf_path,
                RESULT_POSTGRES_DSN,
            ]
        )

        if required_metadata_available:
            try:
                await save_failed_result_to_postgres(
                    check_name=str(check_name),
                    file_id=str(file_id),
                    topic_id=int(topic_id_raw),
                    subject_id=str(subject_id),
                    file_name=str(file_name),
                    input_pdf_path=str(input_pdf_path),
                    error_message=str(exc),
                )
            except Exception as database_error:
                print(
                    "Cannot save failed result to PostgreSQL: "
                    f"{database_error}",
                    flush=True,
                )
        else:
            print(
                "Cannot save failed result because worker "
                "metadata is incomplete",
                flush=True,
            )

        print(
            f"=== Plagiarism worker failed: {exc} ===",
            flush=True,
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())