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

# Chỉ bảo vệ STATE trong một process Uvicorn.
# Stage 1 phải chạy 1 replica và 1 Uvicorn worker cho mỗi check.
PART_RESULT_LOCK = asyncio.Lock()


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
PLAGIARISM_CONCLUSION_THRESHOLD = 0.8


async def shutdown_after_final_written(delay_seconds: int = 2):
    await asyncio.sleep(delay_seconds)

    print(
        "=== Final result saved, shutting down preprocess server ===",
        flush=True,
    )

    os.kill(os.getpid(), signal.SIGTERM)


def reset_state() -> None:
    STATE.update(
        {
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

        pipe.delete(redis_error_key(check_name))
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


def normalize_sentences_for_embedding(sentence_records: list[Any]) -> list[dict]:
    sentence_items = [
        sentence_record_to_dict(sentence)
        for sentence in sentence_records
    ]

    sentence_items = sorted(
        sentence_items,
        key=lambda item: int(item.get("sentence_index", 0)),
    )

    for position, sentence in enumerate(sentence_items):
        sentence_index = int(
            sentence.get("sentence_index", position)
        )

        if sentence_index != position:
            raise ValueError(
                "Sentence index mismatch: "
                f"list_position={position}, "
                f"sentence_index={sentence_index}. "
                "Checker requires sentence_index == list position."
            )

        if not sentence.get("sentence_text"):
            raise ValueError(
                "Missing sentence_text at "
                f"sentence_index={sentence_index}"
            )

    return sentence_items


def validate_embeddings_alignment(
    sentence_items: list[dict],
    query_embeddings: list[list[float]],
) -> None:
    if len(sentence_items) != len(query_embeddings):
        raise ValueError(
            "Embedding count mismatch: "
            f"sentences={len(sentence_items)}, "
            f"embeddings={len(query_embeddings)}"
        )

    for position, vector in enumerate(query_embeddings):
        if len(vector) != 768:
            raise ValueError(
                "Embedding dim mismatch at "
                f"sentence_index={position}: "
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


def parse_minio_path(path: str) -> tuple[str, str]:
    if not path.startswith("minio://"):
        raise ValueError(f"Invalid MinIO path: {path}")

    raw = path.replace("minio://", "", 1)

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
                f"Cannot load PDF from MinIO: {path}, error={exc}"
            ) from exc

    return await asyncio.to_thread(_load)


def split_sentence_indices(
    total_sentences: int,
    part_index: int,
    part_count: int,
) -> list[int]:
    if total_sentences < 0:
        raise ValueError("total_sentences must be >= 0")

    if part_count <= 0:
        raise ValueError("part_count must be > 0")

    if part_index < 0 or part_index >= part_count:
        raise ValueError(
            f"Invalid part_index={part_index}, "
            f"part_count={part_count}"
        )

    base = total_sentences // part_count
    remainder = total_sentences % part_count

    start = (
        part_index * base
        + min(part_index, remainder)
    )
    size = base + (1 if part_index < remainder else 0)
    end = start + size

    return list(range(start, end))


def semantic_result_payload(result: dict) -> dict:
    """Loại các trường timing để nhận diện retry cùng kết quả."""
    return {
        "sentence_indices": result.get("sentence_indices", []),
        "sentence_labels": result.get("sentence_labels", []),
        "references": result.get("references", []),
    }


def merge_results() -> dict:
    total_sentences = len(STATE["sentences"])
    expected_parts = int(STATE["expected_parts"])

    final_labels = [0] * total_sentences
    processed_indices: set[int] = set()
    references_by_document: dict[str, dict] = {}

    for part_index in range(expected_parts):
        if part_index not in STATE["part_results"]:
            raise ValueError(
                f"Missing part result: part_index={part_index}"
            )

        part_result = STATE["part_results"][part_index]

        sentence_indices = [
            int(index)
            for index in part_result.get("sentence_indices", [])
        ]
        labels = [
            int(label)
            for label in part_result.get("sentence_labels", [])
        ]

        expected_indices = split_sentence_indices(
            total_sentences=total_sentences,
            part_index=part_index,
            part_count=expected_parts,
        )

        if sentence_indices != expected_indices:
            raise ValueError(
                "Unexpected sentence partition: "
                f"part_index={part_index}, "
                f"expected={expected_indices}, "
                f"received={sentence_indices}"
            )

        if len(sentence_indices) != len(labels):
            raise ValueError(
                "Label count mismatch: "
                f"part_index={part_index}, "
                f"indices={len(sentence_indices)}, "
                f"labels={len(labels)}"
            )

        for global_index, label in zip(sentence_indices, labels):
            if global_index in processed_indices:
                raise ValueError(
                    "Sentence processed by more than one partition: "
                    f"sentence_index={global_index}"
                )

            if global_index < 0 or global_index >= total_sentences:
                raise ValueError(
                    "Invalid global sentence index: "
                    f"{global_index}"
                )

            if label not in (0, 1):
                raise ValueError(
                    "Invalid sentence label: "
                    f"sentence_index={global_index}, label={label}"
                )

            processed_indices.add(global_index)
            final_labels[global_index] = label

        references = part_result.get("references", [])
        if not isinstance(references, list):
            raise ValueError(
                f"Invalid references in part_index={part_index}"
            )

        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError(
                    f"Invalid reference in part_index={part_index}"
                )

            document_id = str(reference["document_id"])

            if document_id not in references_by_document:
                merged_reference = dict(reference)
                merged_reference["document_id"] = document_id
                merged_reference["plagiarized_count"] = 0
                merged_reference["plagiarism_ratio"] = 0.0
                merged_reference["matched_sentences"] = []
                references_by_document[document_id] = merged_reference

            target = references_by_document[document_id]
            matches = reference.get("matched_sentences", [])

            if not isinstance(matches, list):
                raise ValueError(
                    "Invalid matched_sentences for "
                    f"document_id={document_id}"
                )

            target["matched_sentences"].extend(matches)
            target["plagiarized_count"] += len(matches)

            # Giữ nguyên ý nghĩa ratio hiện tại của checker:
            # mỗi pod trả p_count / total_ref_sentences.
            target["plagiarism_ratio"] = round(
                float(target["plagiarism_ratio"])
                + float(reference.get("plagiarism_ratio", 0.0)),
                4,
            )

    expected_indices = set(range(total_sentences))
    if processed_indices != expected_indices:
        missing = sorted(expected_indices - processed_indices)
        unexpected = sorted(processed_indices - expected_indices)

        raise ValueError(
            "Sentence coverage mismatch: "
            f"missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )

    references = list(references_by_document.values())

    for reference in references:
        reference["matched_sentences"].sort(
            key=lambda item: int(
                item.get("query_sentence_index", -1)
            )
        )

    candidate_order = {
        str(candidate["document_id"]): order
        for order, candidate in enumerate(STATE["candidates"])
    }

    references.sort(
        key=lambda reference: candidate_order.get(
            str(reference["document_id"]),
            len(candidate_order),
        )
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
        "is_plagiarized": (
            plagiarism_ratio > PLAGIARISM_CONCLUSION_THRESHOLD
        ),
        "sentence_labels": final_labels,
        "references": references,
    }


async def prepare_data():
    reset_state()
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

        if expected_parts <= 0:
            raise ValueError("EXPECTED_PARTS must be > 0")

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

        print(
            f"Loaded input file bytes: {len(pdf_bytes)}",
            flush=True,
        )

        full_text, sentence_records = await asyncio.to_thread(
            preprocessing.extract_and_preprocess,
            BytesIO(pdf_bytes),
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

        print(
            f"Extracted sentences: {len(sentence_items)}",
            flush=True,
        )
        print(
            "Generating query embeddings in stage1...",
            flush=True,
        )

        query_embeddings = await asyncio.to_thread(
            embedding.embed_sentences,
            sentence_texts,
        )

        validate_embeddings_alignment(
            sentence_items=sentence_items,
            query_embeddings=query_embeddings,
        )

        print(
            "Generated query embeddings in stage1: "
            f"sentences={len(sentence_items)}, "
            f"embeddings={len(query_embeddings)}",
            flush=True,
        )

        minhash_values = await asyncio.to_thread(
            minhash.compute_minhash,
            full_text,
        )

        connection = await asyncpg.connect(POSTGRES_DSN)

        try:
            candidates = await minhash.find_candidates_by_minhash(
                conn=connection,
                minhash=minhash_values,
                subject_id=subject_id,
                threshold=MINHASH_THRESHOLD,
            )
        finally:
            await connection.close()

        candidates = [
            dict(candidate)
            if not isinstance(candidate, dict)
            else candidate
            for candidate in candidates
        ]

        print(
            f"Candidates found: {len(candidates)}",
            flush=True,
        )

        STATE["full_text"] = full_text
        STATE["sentences"] = sentence_items
        STATE["query_embeddings"] = query_embeddings
        STATE["candidates"] = candidates
        STATE["ready"] = True

        print(
            "=== Stage 1 preprocess server READY ===",
            flush=True,
        )

    except Exception as exc:
        STATE["ready"] = False
        STATE["failed"] = True
        STATE["error"] = str(exc)

        await save_error_to_redis(check_name, str(exc))

        print(
            f"=== Stage 1 preprocess server FAILED: {exc} ===",
            flush=True,
        )


@app.on_event("startup")
async def startup_event():
    app.state.prepare_task = asyncio.create_task(prepare_data())


@app.get("/health")
def health():
    return {"status": "alive"}


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
        "expected_parts": STATE["expected_parts"],
    }


@app.get("/part")
def get_part(part_index: int, part_count: int):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail="preprocess not ready",
        )

    total_sentences = len(STATE["sentences"])
    total_embeddings = len(STATE["query_embeddings"])

    if total_sentences != total_embeddings:
        raise HTTPException(
            status_code=500,
            detail=(
                "Embedding mismatch: "
                f"sentences={total_sentences}, "
                f"embeddings={total_embeddings}"
            ),
        )

    expected_parts = int(STATE["expected_parts"])

    if part_count != expected_parts:
        raise HTTPException(
            status_code=400,
            detail=(
                "part_count mismatch: "
                f"requested={part_count}, "
                f"expected={expected_parts}"
            ),
        )

    try:
        sentence_indices = split_sentence_indices(
            total_sentences=total_sentences,
            part_index=part_index,
            part_count=part_count,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    sentences_slice = [
        STATE["sentences"][index]
        for index in sentence_indices
    ]
    embeddings_slice = [
        STATE["query_embeddings"][index]
        for index in sentence_indices
    ]

    if len(sentences_slice) != len(embeddings_slice):
        raise HTTPException(
            status_code=500,
            detail=(
                "Partition embedding mismatch: "
                f"sentences={len(sentences_slice)}, "
                f"embeddings={len(embeddings_slice)}"
            ),
        )

    return {
        "subject_id": STATE["subject_id"],
        "part_index": part_index,
        "part_count": part_count,
        "total_sentences": total_sentences,
        "partition_sentence_count": len(sentence_indices),
        "sentence_indices": sentence_indices,
        "sentences": sentences_slice,
        "query_embeddings": embeddings_slice,
        # Mọi pod Stage 2 nhận toàn bộ candidates, cùng thứ tự.
        "total_candidates": len(STATE["candidates"]),
        "candidates": STATE["candidates"],
    }


def parse_optional_int(
    value: Any,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: {value}",
        ) from exc


@app.post("/part-result")
async def part_result(payload: dict):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail="preprocess not ready",
        )

    raw_part_index = payload.get("part_index")
    result = payload.get("result")

    if raw_part_index is None:
        raise HTTPException(
            status_code=400,
            detail="Missing part_index",
        )

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid result",
        )

    part_index = parse_optional_int(
        raw_part_index,
        "part_index",
    )
    assert part_index is not None

    expected_parts = int(STATE["expected_parts"])

    if expected_parts <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid expected_parts={expected_parts}",
        )

    if part_index < 0 or part_index >= expected_parts:
        raise HTTPException(
            status_code=400,
            detail=(
                "part_index out of range: "
                f"part_index={part_index}, "
                f"expected_parts={expected_parts}"
            ),
        )

    total_sentences = len(STATE["sentences"])
    expected_sentence_indices = split_sentence_indices(
        total_sentences=total_sentences,
        part_index=part_index,
        part_count=expected_parts,
    )

    raw_sentence_indices = result.get("sentence_indices")
    raw_sentence_labels = result.get("sentence_labels")

    if not isinstance(raw_sentence_indices, list):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid result.sentence_indices",
        )

    if not isinstance(raw_sentence_labels, list):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid result.sentence_labels",
        )

    try:
        sentence_indices = [
            int(index)
            for index in raw_sentence_indices
        ]
        sentence_labels = [
            int(label)
            for label in raw_sentence_labels
        ]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "sentence_indices and sentence_labels "
                "must contain integers"
            ),
        ) from exc

    if sentence_indices != expected_sentence_indices:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Incorrect sentence partition",
                "part_index": part_index,
                "expected_count": len(expected_sentence_indices),
                "received_count": len(sentence_indices),
                "expected_first_indices": (
                    expected_sentence_indices[:10]
                ),
                "received_first_indices": sentence_indices[:10],
            },
        )

    if len(sentence_indices) != len(sentence_labels):
        raise HTTPException(
            status_code=400,
            detail=(
                "sentence_labels length mismatch: "
                f"indices={len(sentence_indices)}, "
                f"labels={len(sentence_labels)}"
            ),
        )

    invalid_labels = [
        {
            "local_index": local_index,
            "label": label,
        }
        for local_index, label in enumerate(sentence_labels)
        if label not in (0, 1)
    ]

    if invalid_labels:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "sentence_labels must contain only 0 or 1"
                ),
                "invalid_labels": invalid_labels[:10],
            },
        )

    result_part_index = parse_optional_int(
        result.get("part_index"),
        "result.part_index",
    )
    if (
        result_part_index is not None
        and result_part_index != part_index
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "part_index mismatch between payload and result: "
                f"payload={part_index}, "
                f"result={result_part_index}"
            ),
        )

    result_part_count = parse_optional_int(
        result.get("part_count"),
        "result.part_count",
    )
    if (
        result_part_count is not None
        and result_part_count != expected_parts
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "part_count mismatch: "
                f"received={result_part_count}, "
                f"expected={expected_parts}"
            ),
        )

    global_total_sentences = parse_optional_int(
        result.get("global_total_sentences"),
        "result.global_total_sentences",
    )
    if (
        global_total_sentences is not None
        and global_total_sentences != total_sentences
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "global_total_sentences mismatch: "
                f"received={global_total_sentences}, "
                f"expected={total_sentences}"
            ),
        )

    references = result.get("references", [])
    if not isinstance(references, list):
        raise HTTPException(
            status_code=400,
            detail="result.references must be a list",
        )

    normalized_result = dict(result)
    normalized_result["part_index"] = part_index
    normalized_result["part_count"] = expected_parts
    normalized_result["sentence_indices"] = sentence_indices
    normalized_result["sentence_labels"] = sentence_labels
    normalized_result["partition_sentence_count"] = len(
        sentence_indices
    )
    normalized_result["global_total_sentences"] = total_sentences
    normalized_result["references"] = references

    async with PART_RESULT_LOCK:
        existing_result = STATE["part_results"].get(part_index)
        duplicate = existing_result is not None

        if existing_result is not None:
            if semantic_result_payload(
                existing_result
            ) != semantic_result_payload(normalized_result):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A different semantic result already exists "
                        f"for part_index={part_index}"
                    ),
                )

            print(
                "Duplicate part result accepted: "
                f"part_index={part_index}",
                flush=True,
            )
        else:
            STATE["part_results"][part_index] = normalized_result

        received = len(STATE["part_results"])

        print(
            "Received sentence partition result: "
            f"part_index={part_index}, "
            f"sentences={len(sentence_indices)}, "
            f"received={received}/{expected_parts}",
            flush=True,
        )

        if received < expected_parts:
            return {
                "ok": True,
                "duplicate": duplicate,
                "part_index": part_index,
                "received": received,
                "expected": expected_parts,
                "final_written": False,
            }

        if STATE["final_written"]:
            return {
                "ok": True,
                "duplicate": duplicate,
                "part_index": part_index,
                "received": received,
                "expected": expected_parts,
                "final_written": True,
            }

        check_name = STATE["check_name"]

        try:
            final_result = merge_results()
            await save_json_to_redis(check_name, final_result)

            STATE["final_written"] = True
            STATE["failed"] = False
            STATE["error"] = None

            print(
                "=== Final sentence-based result saved to Redis ===",
                flush=True,
            )

            asyncio.create_task(
                shutdown_after_final_written(delay_seconds=2)
            )

        except Exception as exc:
            STATE["failed"] = True
            STATE["error"] = str(exc)

            await save_error_to_redis(check_name, str(exc))

            print(
                "=== Failed to merge or save final result: "
                f"{exc} ===",
                flush=True,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Cannot merge or save final result: "
                    f"{exc}"
                ),
            ) from exc

        return {
            "ok": True,
            "duplicate": duplicate,
            "part_index": part_index,
            "received": received,
            "expected": expected_parts,
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
    # Không chạy nhiều Uvicorn worker vì STATE và Lock nằm trong memory.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )