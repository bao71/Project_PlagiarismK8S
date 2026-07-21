import asyncio
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from minio import Minio
from minio.error import S3Error
from pymilvus import MilvusClient

from app.services import embedding
from app.services import minhash
from app.services import preprocessing
from app.services.preprocessing import SentenceRecord


router = APIRouter(
    prefix="/reference-documents",
    tags=["Reference Documents"],
)


# =====================================================
# PostgreSQL
# =====================================================

POSTGRES_DSN = os.getenv(
    "MINHASH_POSTGRES_DSN",
    os.getenv(
        "POSTGRES_DSN",
        (
            "postgresql://postgres:postgres"
            "@postgresdb.streaming.svc.cluster.local:5432/plagiarism"
        ),
    ),
)


# =====================================================
# MinIO
# =====================================================

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

REFERENCE_BUCKET = os.getenv(
    "REFERENCE_BUCKET",
    "reference-documents",
)


# =====================================================
# Milvus
# =====================================================

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_HOST = os.getenv(
    "MILVUS_HOST",
    "milvus-cluster.milvus-cluster.svc.cluster.local",
)
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

MILVUS_COLLECTION_NAME = os.getenv(
    "MILVUS_COLLECTION_NAME",
    "PlagiarismDetection",
)

MILVUS_BATCH_SIZE = int(
    os.getenv("MILVUS_BATCH_SIZE", "200")
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "768")
)


# =====================================================
# Upload validation
# =====================================================

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_BYTES",
        str(50 * 1024 * 1024),
    )
)

ALLOWED_FILE_EXTENSIONS = {
    ".pdf",
    ".docx",
}


def get_milvus_uri() -> str:
    if MILVUS_URI:
        return MILVUS_URI

    return f"http://{MILVUS_HOST}:{MILVUS_PORT}"


def get_minio_client() -> Minio:
    if not MINIO_ACCESS_KEY:
        raise RuntimeError("Missing MINIO_ACCESS_KEY")

    if not MINIO_SECRET_KEY:
        raise RuntimeError("Missing MINIO_SECRET_KEY")

    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def get_milvus_client() -> MilvusClient:
    return MilvusClient(
        uri=get_milvus_uri(),
        timeout=60,
    )


def normalize_file_name(file_name: str | None) -> str:
    if not file_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing uploaded file name",
        )

    # Bỏ toàn bộ phần thư mục mà client có thể gửi lên.
    normalized = Path(file_name).name.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid uploaded file name",
        )

    extension = Path(normalized).suffix.lower()

    if extension not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only PDF and DOCX files are allowed. "
                f"Received extension: {extension or '<empty>'}"
            ),
        )

    return normalized


def normalize_subject_id(subject_id: str) -> str:
    normalized = subject_id.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="subject_id must not be empty",
        )

    return normalized


def safe_object_path_component(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = value.strip("._")

    return value or "unknown"


def sentence_record_to_dict(sentence: Any) -> dict[str, Any]:
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
        key=lambda item: int(
            item.get("sentence_index", 0)
        )
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

        sentence_text = str(
            sentence.get("sentence_text", "")
        ).strip()

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
                sentence_text=sentence_text,
                bbox_x0=float(
                    sentence.get("bbox_x0", 0.0)
                ),
                bbox_y0=float(
                    sentence.get("bbox_y0", 0.0)
                ),
                bbox_x1=float(
                    sentence.get("bbox_x1", 0.0)
                ),
                bbox_y1=float(
                    sentence.get("bbox_y1", 0.0)
                ),
            )
        )

    return normalized_records


def validate_embeddings_alignment(
    sentence_records: list[SentenceRecord],
    sentence_embeddings: list[list[float]],
) -> None:
    if len(sentence_records) != len(sentence_embeddings):
        raise ValueError(
            "Embedding count mismatch: "
            f"sentences={len(sentence_records)}, "
            f"embeddings={len(sentence_embeddings)}"
        )

    for position, vector in enumerate(sentence_embeddings):
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"sentence_index={position}, "
                f"dimension={len(vector)}, "
                f"expected={EMBEDDING_DIMENSION}"
            )


def normalize_minhash_values(
    values: Any,
) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, (list, tuple)):
        raise TypeError(
            "compute_minhash() must return a list, "
            f"tuple or NumPy array, received={type(values)}"
        )

    normalized = [int(value) for value in values]

    if not normalized:
        raise ValueError("MinHash signature is empty")

    # PostgreSQL integer[] chỉ chứa signed int32.
    min_int32 = -2_147_483_648
    max_int32 = 2_147_483_647

    invalid_values = [
        value
        for value in normalized
        if value < min_int32 or value > max_int32
    ]

    if invalid_values:
        raise ValueError(
            "MinHash value exceeds PostgreSQL integer[] range. "
            "Do not silently truncate the signature. "
            "Update app.services.minhash.compute_minhash() "
            "or change documents.minhash to bigint[]/bytea. "
            f"Example invalid value={invalid_values[0]}"
        )

    return normalized


def upload_bytes_to_minio(
    *,
    object_name: str,
    file_bytes: bytes,
    content_type: str,
) -> str:
    client = get_minio_client()

    try:
        if not client.bucket_exists(REFERENCE_BUCKET):
            client.make_bucket(REFERENCE_BUCKET)

        client.put_object(
            bucket_name=REFERENCE_BUCKET,
            object_name=object_name,
            data=BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )

    except S3Error as exc:
        raise RuntimeError(
            "Cannot upload reference document to MinIO: "
            f"bucket={REFERENCE_BUCKET}, "
            f"object={object_name}, error={exc}"
        ) from exc

    return f"minio://{REFERENCE_BUCKET}/{object_name}"


def remove_minio_object(object_name: str) -> None:
    try:
        client = get_minio_client()
        client.remove_object(
            bucket_name=REFERENCE_BUCKET,
            object_name=object_name,
        )
    except Exception as exc:
        print(
            "Cleanup warning: cannot remove MinIO object: "
            f"bucket={REFERENCE_BUCKET}, "
            f"object={object_name}, error={exc}",
            flush=True,
        )


async def insert_document_to_postgres(
    *,
    document_id: UUID,
    file_name: str,
    subject_id: str,
    file_path: str,
    minhash_values: list[int],
) -> None:
    connection = await asyncpg.connect(POSTGRES_DSN)

    try:
        await connection.execute(
            """
            INSERT INTO public.documents (
                id,
                file_name,
                subject_id,
                file_path,
                minhash
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            document_id,
            file_name,
            subject_id,
            file_path,
            minhash_values,
        )
    finally:
        await connection.close()


async def delete_document_from_postgres(
    document_id: UUID,
) -> None:
    connection = await asyncpg.connect(POSTGRES_DSN)

    try:
        await connection.execute(
            """
            DELETE FROM public.documents
            WHERE id = $1
            """,
            document_id,
        )
    finally:
        await connection.close()


def build_milvus_rows(
    *,
    document_id: UUID,
    file_name: str,
    subject_id: str,
    sentence_records: list[SentenceRecord],
    sentence_embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    document_id_text = str(document_id)
    rows: list[dict[str, Any]] = []

    for sentence, vector in zip(
        sentence_records,
        sentence_embeddings,
    ):
        rows.append(
            {
                # Primary key của từng câu trong Milvus.
                "id": str(uuid4()),

                # UUID của document trong PostgreSQL.
                "document_id": document_id_text,

                "file_name": file_name,
                "subject_id": subject_id,

                "sentence_index": int(
                    sentence.sentence_index
                ),
                "sentence_index_page": int(
                    sentence.sentence_index_page
                ),
                "page_number": int(
                    sentence.page_number
                ),
                "sentence_text": sentence.sentence_text,

                "bbox_x0": float(sentence.bbox_x0),
                "bbox_y0": float(sentence.bbox_y0),
                "bbox_x1": float(sentence.bbox_x1),
                "bbox_y1": float(sentence.bbox_y1),

                "embedding": [
                    float(value)
                    for value in vector
                ],
            }
        )

    return rows


def insert_rows_to_milvus(
    rows: list[dict[str, Any]],
) -> int:
    client = get_milvus_client()
    inserted_count = 0

    try:
        for start in range(
            0,
            len(rows),
            MILVUS_BATCH_SIZE,
        ):
            batch = rows[
                start:start + MILVUS_BATCH_SIZE
            ]

            client.insert(
                collection_name=MILVUS_COLLECTION_NAME,
                data=batch,
            )

            inserted_count += len(batch)

        client.flush(
            collection_name=MILVUS_COLLECTION_NAME
        )

        return inserted_count

    finally:
        try:
            client.close()
        except Exception:
            pass


def delete_document_from_milvus(
    document_id: UUID,
) -> None:
    client = get_milvus_client()

    try:
        client.delete(
            collection_name=MILVUS_COLLECTION_NAME,
            filter=(
                f'document_id == "{str(document_id)}"'
            ),
        )

        client.flush(
            collection_name=MILVUS_COLLECTION_NAME
        )

    except Exception as exc:
        print(
            "Cleanup warning: cannot delete document "
            "from Milvus: "
            f"document_id={document_id}, error={exc}",
            flush=True,
        )

    finally:
        try:
            client.close()
        except Exception:
            pass


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "alive",
    }


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_reference_document(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    subject_id = normalize_subject_id(subject_id)
    file_name = normalize_file_name(file.filename)

    file_bytes = await file.read()
    await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Uploaded file is too large: "
                f"bytes={len(file_bytes)}, "
                f"maximum={MAX_UPLOAD_BYTES}"
            ),
        )

    document_id = uuid4()

    object_name = (
        f"{safe_object_path_component(subject_id)}/"
        f"{document_id}/"
        f"{safe_object_path_component(file_name)}"
    )

    minio_uploaded = False
    postgres_inserted = False
    milvus_insert_started = False

    try:
        print(
            "=== Upload reference document starting ===",
            flush=True,
        )
        print(
            f"document_id={document_id}",
            flush=True,
        )
        print(
            f"subject_id={subject_id}",
            flush=True,
        )
        print(
            f"file_name={file_name}",
            flush=True,
        )

        file_path = await asyncio.to_thread(
            upload_bytes_to_minio,
            object_name=object_name,
            file_bytes=file_bytes,
            content_type=(
                file.content_type
                or "application/octet-stream"
            ),
        )

        minio_uploaded = True

        print(
            f"Uploaded original file to MinIO: {file_path}",
            flush=True,
        )

        full_text, raw_sentence_records = (
            await asyncio.to_thread(
                preprocessing.extract_and_preprocess,
                BytesIO(file_bytes),
            )
        )

        if not full_text or not full_text.strip():
            raise ValueError(
                "Cannot extract full_text from uploaded document"
            )

        if not raw_sentence_records:
            raise ValueError(
                "Cannot extract any sentence "
                "from uploaded document"
            )

        sentence_records = normalize_sentence_records(
            raw_sentence_records
        )

        print(
            "Preprocess completed: "
            f"sentences={len(sentence_records)}",
            flush=True,
        )

        raw_minhash_values = await asyncio.to_thread(
            minhash.compute_minhash,
            full_text,
        )

        minhash_values = normalize_minhash_values(
            raw_minhash_values
        )

        print(
            "MinHash generated: "
            f"length={len(minhash_values)}",
            flush=True,
        )

        sentence_texts = [
            sentence.sentence_text
            for sentence in sentence_records
        ]

        sentence_embeddings = await asyncio.to_thread(
            embedding.embed_sentences,
            sentence_texts,
        )

        validate_embeddings_alignment(
            sentence_records=sentence_records,
            sentence_embeddings=sentence_embeddings,
        )

        print(
            "Embeddings generated: "
            f"count={len(sentence_embeddings)}, "
            f"dimension={EMBEDDING_DIMENSION}",
            flush=True,
        )

        await insert_document_to_postgres(
            document_id=document_id,
            file_name=file_name,
            subject_id=subject_id,
            file_path=file_path,
            minhash_values=minhash_values,
        )

        postgres_inserted = True

        print(
            "Saved document metadata and MinHash "
            "to PostgreSQL",
            flush=True,
        )

        milvus_rows = build_milvus_rows(
            document_id=document_id,
            file_name=file_name,
            subject_id=subject_id,
            sentence_records=sentence_records,
            sentence_embeddings=sentence_embeddings,
        )

        milvus_insert_started = True

        inserted_vectors = await asyncio.to_thread(
            insert_rows_to_milvus,
            milvus_rows,
        )

        print(
            "Saved sentence embeddings to Milvus: "
            f"count={inserted_vectors}",
            flush=True,
        )

        print(
            "=== Upload reference document completed ===",
            flush=True,
        )

        return {
            "document_id": str(document_id),
            "file_name": file_name,
            "subject_id": subject_id,
            "file_path": file_path,
            "minhash_length": len(minhash_values),
            "sentence_count": len(sentence_records),
            "milvus_vector_count": inserted_vectors,
            "status": "completed",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "=== Upload reference document failed ===",
            flush=True,
        )
        print(
            f"document_id={document_id}, error={exc}",
            flush=True,
        )

        # Milvus có thể insert thành công một phần trước khi lỗi.
        if milvus_insert_started:
            await asyncio.to_thread(
                delete_document_from_milvus,
                document_id,
            )

        if postgres_inserted:
            try:
                await delete_document_from_postgres(
                    document_id
                )
            except Exception as cleanup_error:
                print(
                    "Cleanup warning: cannot delete "
                    "PostgreSQL document: "
                    f"document_id={document_id}, "
                    f"error={cleanup_error}",
                    flush=True,
                )

        if minio_uploaded:
            await asyncio.to_thread(
                remove_minio_object,
                object_name,
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Cannot index reference document: "
                f"{exc}"
            ),
        ) from exc