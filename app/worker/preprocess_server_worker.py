import asyncio
import json
import os
from io import BytesIO
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from minio import Minio
from minio.error import S3Error
import uvicorn

from app.services import preprocessing
from app.services import minhash


app = FastAPI()

STATE = {
    "ready": False,
    "failed": False,
    "error": None,

    "subject_id": None,
    "input_pdf_path": None,
    "result_output_path": None,

    "full_text": None,
    "sentences": [],
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

MINHASH_THRESHOLD = float(
    os.getenv("MINHASH_THRESHOLD", "0.05")
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


async def save_json_to_minio(path: str, data: dict):
    bucket, object_name = parse_minio_path(path)
    client = create_minio_client()

    payload = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        default=str,
    ).encode("utf-8")

    try:
        if not client.bucket_exists(bucket):
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
    results = []

    for s in sentences:
        results.append({
            "page_number": getattr(s, "page_number", 0),
            "sentence_index": getattr(s, "sentence_index", 0),
            "sentence_index_page": getattr(s, "sentence_index_page", 0),
            "sentence_text": getattr(s, "sentence_text", ""),
            "bbox_x0": getattr(s, "bbox_x0", 0.0),
            "bbox_y0": getattr(s, "bbox_y0", 0.0),
            "bbox_x1": getattr(s, "bbox_x1", 0.0),
            "bbox_y1": getattr(s, "bbox_y1", 0.0),
        })

    return results


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
            if label == 1:
                final_labels[i] = 1

        references.extend(part_result.get("references", []))

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
    try:
        subject_id = os.getenv("SUBJECT_ID")
        input_pdf_path = os.getenv("INPUT_PDF_PATH")
        result_output_path = os.getenv("RESULT_OUTPUT_PATH")
        expected_parts = int(os.getenv("EXPECTED_PARTS", "1"))

        if not subject_id:
            raise ValueError("Missing SUBJECT_ID")

        if not input_pdf_path:
            raise ValueError("Missing INPUT_PDF_PATH")

        if not result_output_path:
            raise ValueError("Missing RESULT_OUTPUT_PATH")

        if not POSTGRES_DSN:
            raise ValueError("Missing POSTGRES_DSN")

        STATE["subject_id"] = subject_id
        STATE["input_pdf_path"] = input_pdf_path
        STATE["result_output_path"] = result_output_path
        STATE["expected_parts"] = expected_parts

        print("=== Stage 1 preprocess server starting ===", flush=True)
        print(f"SUBJECT_ID={subject_id}", flush=True)
        print(f"INPUT_PDF_PATH={input_pdf_path}", flush=True)
        print(f"RESULT_OUTPUT_PATH={result_output_path}", flush=True)
        print(f"EXPECTED_PARTS={expected_parts}", flush=True)

        pdf_bytes = await load_pdf_bytes_from_minio(input_pdf_path)

        if not pdf_bytes:
            raise ValueError("PDF file is empty")

        print(f"Loaded PDF bytes: {len(pdf_bytes)}", flush=True)

        full_text, sentence_records = preprocessing.extract_and_preprocess(
            BytesIO(pdf_bytes)
        )

        if not full_text:
            raise ValueError("Cannot extract full_text")

        if not sentence_records:
            raise ValueError("Cannot extract any sentence")

        sentences = convert_sentences(sentence_records)

        print(f"Extracted sentences: {len(sentences)}", flush=True)

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
        STATE["sentences"] = sentences
        STATE["candidates"] = candidates

        STATE["ready"] = True

        print("=== Stage 1 preprocess server READY ===", flush=True)

    except Exception as e:
        STATE["failed"] = True
        STATE["error"] = str(e)

        print(f"=== Stage 1 preprocess server FAILED: {e} ===", flush=True)


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

    candidates_slice = split_candidates(
        candidates=STATE["candidates"],
        part_index=part_index,
        part_count=part_count,
    )

    return {
        "subject_id": STATE["subject_id"],
        "part_index": part_index,
        "part_count": part_count,
        "total_candidates": len(STATE["candidates"]),
        "processed_candidates": len(candidates_slice),
        "total_sentences": len(STATE["sentences"]),
        "sentences": STATE["sentences"],
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

        await save_json_to_minio(
            STATE["result_output_path"],
            final_result,
        )

        STATE["final_written"] = True

        print("=== Final result written ===", flush=True)

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
        "candidate_count": len(STATE["candidates"]),
        "sentence_count": len(STATE["sentences"]),
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