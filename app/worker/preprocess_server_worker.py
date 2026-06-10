import asyncio
import json
import os
import signal
from io import BytesIO
from typing import Any

import asyncpg
import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from minio import Minio
from minio.error import S3Error

from app.services import embedding
from app.services import minhash
from app.services import preprocessing


app = FastAPI()


STATE = {
    "ready": False,
    "failed": False,
    "error": None,

    "check_name": None,
    "subject_id": None,
    "input_pdf_path": None,

    "full_text": None,
    "sentences": [],
    "query_embeddings": [],
    "candidates": [],

    "expected_parts": 1,
    "part_results": {},

    "final_written": False,
}


MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

POSTGRES_DSN = os.getenv("POSTGRES_DSN")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "86400"))

MINHASH_THRESHOLD = float(
    os.getenv("MINHASH_THRESHOLD", "0.05")
)


async def shutdown_after_final_written(delay_seconds: int = 2):
    await asyncio.sleep(delay_seconds)

    print(
        "=== Final result saved, shutting down preprocess server ===",
        flush=True,
    )

    os.kill(
        os.getpid(),
        signal.SIGTERM,
    )


def redis_status_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:status"


def redis_result_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:result"


def redis_error_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:error"


def create_redis_client() -> redis.Redis:
    if not REDIS_HOST:
        raise ValueError("Missing REDIS_HOST")

    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=5,
    )

    client.ping()
    return client


async def save_json_to_redis(check_name: str, data: dict):
    def _save():
        client = create_redis_client()
        pipe = client.pipeline()

        pipe.setex(
            redis_result_key(check_name),
            REDIS_TTL_SECONDS,
            json.dumps(
                data,
                ensure_ascii=False,
                default=str,
            ),
        )

        pipe.setex(
            redis_status_key(check_name),
            REDIS_TTL_SECONDS,
            "completed",
        )

        pipe.execute()

    await asyncio.to_thread(_save)

    print(
        f"Saved final result to Redis: check_name={check_name}",
        flush=True,
    )


async def save_error_to_redis(check_name: str | None, error: str):
    if not check_name:
        return

    def _save():
        client = create_redis_client()
        pipe = client.pipeline()

        pipe.setex(
            redis_error_key(check_name),
            REDIS_TTL_SECONDS,
            error,
        )

        pipe.setex(
            redis_status_key(check_name),
            REDIS_TTL_SECONDS,
            "failed",
        )

        pipe.execute()

    try:
        await asyncio.to_thread(_save)
        print(
            f"Saved error to Redis: check_name={check_name}, error={error}",
            flush=True,
        )
    except Exception as redis_error:
        print(
            f"Cannot save error to Redis: check_name={check_name}, "
            f"error={error}, redis_error={redis_error}",
            flush=True,
        )


def sentence_record_to_dict(sentence):
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


def normalize_sentences_for_embedding(sentence_records):
    sentence_items = [
        sentence_record_to_dict(sentence)
        for sentence in sentence_records
    ]

    sentence_items = sorted(
        sentence_items,
        key=lambda x: int(x.get("sentence_index", 0)),
    )

    for pos, sentence in enumerate(sentence_items):
        sentence_index = int(sentence.get("sentence_index", pos))

        if sentence_index != pos:
            raise ValueError(
                f"Sentence index mismatch: "
                f"list_position={pos}, sentence_index={sentence_index}. "
                f"Checker requires sentence_index == list position."
            )

        if not sentence.get("sentence_text"):
            raise ValueError(
                f"Missing sentence_text at sentence_index={sentence_index}"
            )

    return sentence_items


def validate_embeddings_alignment(
    sentence_items: list[dict],
    query_embeddings: list[list[float]],
):
    if len(sentence_items) != len(query_embeddings):
        raise ValueError(
            f"Embedding count mismatch: "
            f"sentences={len(sentence_items)}, "
            f"embeddings={len(query_embeddings)}"
        )

    for pos, vector in enumerate(query_embeddings):
        if len(vector) != 768:
            raise ValueError(
                f"Embedding dim mismatch at sentence_index={pos}: "
                f"dim={len(vector)}, expected=768"
            )


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


def parse_minio_path(path: str):
    if not path.startswith("minio://"):
        raise ValueError(f"Invalid MinIO path: {path}")

    raw = path.replace("minio://", "", 1)

    if "/" not in raw:
        raise ValueError(f"Invalid MinIO path: {path}")

    bucket, object_name = raw.split("/", 1)

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


def split_candidates(
    candidates: list[dict],
    part_index: int,
    part_count: int,
) -> list[dict]:
    if part_count <= 0:
        raise ValueError("part_count must be > 0")

    if part_index < 0 or part_index >= part_count:
        raise ValueError(
            f"Invalid part_index={part_index}, part_count={part_count}"
        )

    total = len(candidates)

    base = total // part_count
    remainder = total % part_count

    start = part_index * base + min(part_index, remainder)
    size = base + (1 if part_index < remainder else 0)
    end = start + size

    return candidates[start:end]


def merge_results() -> dict:
    total_sentences = len(STATE["sentences"])
    final_labels = [0] * total_sentences
    references = []

    for part_index, part_result in STATE["part_results"].items():
        labels = part_result.get("sentence_labels", [])

        for i, label in enumerate(labels):
            if i >= total_sentences:
                break

            if label == 1:
                final_labels[i] = 1

        references.extend(
            part_result.get("references", [])
        )

    plagiarized_sentences = sum(final_labels)

    plagiarism_ratio = (
        round(plagiarized_sentences / total_sentences, 4)
        if total_sentences > 0
        else 0.0
    )

    return {
        "total_sentences": total_sentences,
        "plagiarized_sentences": plagiarized_sentences,
        "plagiarism_ratio": plagiarism_ratio,
        "is_plagiarized": plagiarism_ratio > 0.8,
        "sentence_labels": final_labels,
        "references": references,
    }


async def prepare_data():
    check_name = os.getenv("CHECK_NAME")

    try:
        subject_id = os.getenv("SUBJECT_ID")
        input_pdf_path = os.getenv("INPUT_PDF_PATH")
        expected_parts = int(os.getenv("EXPECTED_PARTS", "1"))

        if not check_name:
            raise ValueError("Missing CHECK_NAME")

        if not subject_id:
            raise ValueError("Missing SUBJECT_ID")

        if not input_pdf_path:
            raise ValueError("Missing INPUT_PDF_PATH")

        if not POSTGRES_DSN:
            raise ValueError("Missing POSTGRES_DSN")

        if not REDIS_HOST:
            raise ValueError("Missing REDIS_HOST")

        STATE["check_name"] = check_name
        STATE["subject_id"] = subject_id
        STATE["input_pdf_path"] = input_pdf_path
        STATE["expected_parts"] = expected_parts

        print("=== Stage 1 preprocess server starting ===", flush=True)
        print(f"CHECK_NAME={check_name}", flush=True)
        print(f"SUBJECT_ID={subject_id}", flush=True)
        print(f"INPUT_PDF_PATH={input_pdf_path}", flush=True)
        print(f"EXPECTED_PARTS={expected_parts}", flush=True)
        print(f"REDIS_HOST={REDIS_HOST}", flush=True)
        print(f"REDIS_PORT={REDIS_PORT}", flush=True)

        pdf_bytes = await load_pdf_bytes_from_minio(input_pdf_path)

        if not pdf_bytes:
            raise ValueError("PDF file is empty")

        print(f"Loaded input file bytes: {len(pdf_bytes)}", flush=True)

        full_text, sentence_records = preprocessing.extract_and_preprocess(
            BytesIO(pdf_bytes)
        )

        if not full_text:
            raise ValueError("Cannot extract full_text")

        if not sentence_records:
            raise ValueError("Cannot extract any sentence")

        sentence_items = normalize_sentences_for_embedding(
            sentence_records
        )

        sentence_texts = [
            sentence["sentence_text"]
            for sentence in sentence_items
        ]

        print(f"Extracted sentences: {len(sentence_items)}", flush=True)
        print("Generating query embeddings in stage1...", flush=True)

        query_embeddings = embedding.embed_sentences(
            sentence_texts
        )

        validate_embeddings_alignment(
            sentence_items=sentence_items,
            query_embeddings=query_embeddings,
        )

        print(
            f"Generated query embeddings in stage1: "
            f"sentences={len(sentence_items)}, "
            f"embeddings={len(query_embeddings)}",
            flush=True,
        )

        minhash_values = minhash.compute_minhash(full_text)

        conn = await asyncpg.connect(POSTGRES_DSN)

        try:
            candidates = await minhash.find_candidates_by_minhash(
                conn=conn,
                minhash=minhash_values,
                subject_id=subject_id,
                threshold=MINHASH_THRESHOLD,
            )

        finally:
            await conn.close()

        print(f"Candidates found: {len(candidates)}", flush=True)

        STATE["full_text"] = full_text
        STATE["sentences"] = sentence_items
        STATE["query_embeddings"] = query_embeddings
        STATE["candidates"] = candidates

        STATE["ready"] = True

        print("=== Stage 1 preprocess server READY ===", flush=True)

    except Exception as e:
        STATE["failed"] = True
        STATE["error"] = str(e)

        await save_error_to_redis(check_name, str(e))

        print(
            f"=== Stage 1 preprocess server FAILED: {e} ===",
            flush=True,
        )


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(prepare_data())


@app.get("/health")
def health():
    return {
        "status": "alive",
    }


@app.get("/ready")
def ready():
    if STATE["failed"]:
        return JSONResponse(
            status_code=500,
            content={
                "ready": False,
                "error": STATE["error"],
            },
        )

    if not STATE["ready"]:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "message": "preprocess not ready",
            },
        )

    return {
        "ready": True,
        "candidate_count": len(STATE["candidates"]),
        "sentence_count": len(STATE["sentences"]),
        "embedding_count": len(STATE["query_embeddings"]),
    }


@app.get("/part")
def get_part(
    part_index: int,
    part_count: int,
):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail="preprocess not ready",
        )

    if len(STATE["sentences"]) != len(STATE["query_embeddings"]):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Embedding mismatch: "
                f"sentences={len(STATE['sentences'])}, "
                f"embeddings={len(STATE['query_embeddings'])}"
            ),
        )

    candidates_slice = split_candidates(
        candidates=STATE["candidates"],
        part_index=part_index,
        part_count=part_count,
    )

    return {
        "subject_id": STATE["subject_id"],
        "part_index": part_index,
        "part_count": part_count,
        "total_sentences": len(STATE["sentences"]),
        "total_candidates": len(STATE["candidates"]),
        "sentences": STATE["sentences"],
        "query_embeddings": STATE["query_embeddings"],
        "candidates": candidates_slice,
    }


@app.post("/part-result")
async def part_result(payload: dict):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail="preprocess not ready",
        )

    part_index = payload.get("part_index")
    result = payload.get("result")

    if part_index is None:
        raise HTTPException(
            status_code=400,
            detail="Missing part_index",
        )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Missing result",
        )

    STATE["part_results"][int(part_index)] = result

    received = len(STATE["part_results"])
    expected = STATE["expected_parts"]

    print(
        f"Received part result: {received}/{expected}",
        flush=True,
    )

    if received == expected and not STATE["final_written"]:
        final_result = merge_results()
        check_name = STATE["check_name"]

        try:
            await save_json_to_redis(
                check_name,
                final_result,
            )

            STATE["final_written"] = True

            print("=== Final result saved to Redis ===", flush=True)

            asyncio.create_task(
                shutdown_after_final_written(delay_seconds=2)
            )

        except Exception as e:
            STATE["failed"] = True
            STATE["error"] = str(e)

            await save_error_to_redis(
                check_name,
                str(e),
            )

            print(
                f"=== Failed to save final result to Redis: {e} ===",
                flush=True,
            )

            raise HTTPException(
                status_code=500,
                detail=f"Cannot save final result to Redis: {e}",
            )

    return {
        "ok": True,
        "received": received,
        "expected": expected,
        "final_written": STATE["final_written"],
    }


@app.get("/status")
def status():
    return {
        "ready": STATE["ready"],
        "failed": STATE["failed"],
        "error": STATE["error"],
        "check_name": STATE["check_name"],
        "candidate_count": len(STATE["candidates"]),
        "sentence_count": len(STATE["sentences"]),
        "embedding_count": len(STATE["query_embeddings"]),
        "received_parts": len(STATE["part_results"]),
        "expected_parts": STATE["expected_parts"],
        "final_written": STATE["final_written"],
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
