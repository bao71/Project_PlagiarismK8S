import json
import os
import re
from typing import Any
from uuid import uuid4

import asyncpg
from fastapi import APIRouter, HTTPException
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel


router = APIRouter(
    prefix="/plagiarism-checks",
    tags=["Plagiarism Checks"],
)


class SubmitCheckRequest(BaseModel):
    fileId: str
    topicId: int
    subjectId: str
    originalFilePath: str
    fileName: str | None = None
    checkName: str | None = None


# =====================================================
# Kubernetes config
# =====================================================

K8S_GROUP = os.getenv("PLAGIARISM_CRD_GROUP", "plagiarism.io")
K8S_VERSION = os.getenv("PLAGIARISM_CRD_VERSION", "v1")
K8S_PLURAL = os.getenv("PLAGIARISM_CRD_PLURAL", "plagiarismchecks")
K8S_NAMESPACE = os.getenv("PLAGIARISM_NAMESPACE", "plagiarism")
K8S_KIND = os.getenv("PLAGIARISM_CRD_KIND", "PlagiarismCheck")


# =====================================================
# MinIO config
# =====================================================

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

VALIDATE_MINIO_OBJECT = (
    os.getenv("VALIDATE_MINIO_OBJECT", "true").lower() == "true"
)


# =====================================================
# PostgreSQL result database
# =====================================================

RESULT_POSTGRES_DSN = os.getenv(
    "RESULT_POSTGRES_DSN",
    (
        "postgresql://plagiarism_user:postgres"
        "@postgresdb.streaming.svc.cluster.local:5432/plagiarism_system"
    ),
)

ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx"}


# =====================================================
# Clients and validation
# =====================================================

def load_k8s_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def get_minio_client() -> Minio:
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def normalize_check_name(check_name: str | None = None) -> str:
    if not check_name:
        return f"check-{uuid4().hex[:8]}"

    name = check_name.lower().strip()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")

    if not name:
        return f"check-{uuid4().hex[:8]}"

    return name[:63]


def is_allowed_document(object_name: str) -> bool:
    lowered = object_name.lower()
    return any(lowered.endswith(ext) for ext in ALLOWED_FILE_EXTENSIONS)


def parse_minio_uri(minio_uri: str) -> tuple[str, str]:
    if not minio_uri:
        raise HTTPException(
            status_code=400,
            detail="originalFilePath is required",
        )

    minio_uri = minio_uri.strip()

    if not minio_uri.startswith("minio://"):
        raise HTTPException(
            status_code=400,
            detail="originalFilePath must start with minio://",
        )

    path = minio_uri.removeprefix("minio://")
    parts = path.split("/", 1)

    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid MinIO path. "
                "Expected format: minio://bucket/object"
            ),
        )

    bucket_name = parts[0].strip()
    object_name = parts[1].strip()

    if not bucket_name or not object_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid MinIO path. Bucket or object name is empty",
        )

    if not is_allowed_document(object_name):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed",
        )

    return bucket_name, object_name


def get_file_name_from_minio_uri(minio_uri: str) -> str:
    _, object_name = parse_minio_uri(minio_uri)
    return object_name.rsplit("/", 1)[-1]


def validate_minio_object_exists(minio_uri: str) -> None:
    bucket_name, object_name = parse_minio_uri(minio_uri)

    if not VALIDATE_MINIO_OBJECT:
        return

    minio_client = get_minio_client()

    try:
        minio_client.stat_object(
            bucket_name=bucket_name,
            object_name=object_name,
        )
    except S3Error as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot access MinIO object: {minio_uri}. "
                f"Error: {exc}"
            ),
        ) from exc


def parse_result(value: Any) -> Any:
    """Parse result từ JSON string thành Python object."""
    if value is None or not isinstance(value, str):
        return value

    parsed: Any = value

    # Hỗ trợ trường hợp result bị JSON encode tối đa hai lần.
    for _ in range(2):
        if not isinstance(parsed, str):
            break

        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return value

    return parsed


# =====================================================
# PostgreSQL helpers
# =====================================================




async def delete_check_record(check_name: str) -> None:
    conn = await asyncpg.connect(RESULT_POSTGRES_DSN)

    try:
        await conn.execute(
            """
            DELETE FROM plagiarism_checks
            WHERE check_name = $1
              AND status = 'queued'
            """,
            check_name,
        )
    finally:
        await conn.close()


async def get_check_record(check_name: str) -> asyncpg.Record | None:
    conn = await asyncpg.connect(RESULT_POSTGRES_DSN)

    try:
        return await conn.fetchrow(
            """
            SELECT
                check_name,
                file_id,
                topic_id,
                subject_id,
                file_name,
                input_pdf_path,
                status,
                result,
                error_message,
                candidate_count,
                total_sentences,
                plagiarized_sentences,
                plagiarism_ratio,
                is_plagiarized,
                created_at,
                started_at,
                completed_at,
                updated_at
            FROM plagiarism_checks
            WHERE check_name = $1
            """,
            check_name,
        )
    finally:
        await conn.close()


# =====================================================
# Kubernetes CR helpers
# =====================================================

def build_cr_body(
    *,
    check_name: str,
    file_id: str,
    topic_id: int,
    subject_id: str,
    file_name: str,
    original_file_path: str,
) -> dict[str, Any]:
    return {
        "apiVersion": f"{K8S_GROUP}/{K8S_VERSION}",
        "kind": K8S_KIND,
        "metadata": {
            "name": check_name,
            "namespace": K8S_NAMESPACE,
        },
        "spec": {
            "fileId": file_id,
            "topicId": topic_id,
            "subjectId": subject_id,
            "fileName": file_name,
            "originalFilePath": original_file_path,
        },
    }


def create_plagiarism_cr(cr_body: dict[str, Any]) -> None:
    load_k8s_config()
    api = client.CustomObjectsApi()

    try:
        api.create_namespaced_custom_object(
            group=K8S_GROUP,
            version=K8S_VERSION,
            namespace=K8S_NAMESPACE,
            plural=K8S_PLURAL,
            body=cr_body,
        )
    except ApiException as exc:
        if exc.status == 409:
            raise HTTPException(
                status_code=409,
                detail=(
                    "PlagiarismCheck CR already exists: "
                    f"{cr_body['metadata']['name']}"
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail=(
                "Cannot create PlagiarismCheck CR: "
                f"{exc.reason}. Body: {exc.body}"
            ),
        ) from exc


# =====================================================
# API endpoints
# =====================================================
@router.post("/submit")
async def submit_from_minio(req: SubmitCheckRequest):
    check_name = normalize_check_name(req.checkName)
    subject_id = str(req.subjectId)
    original_file_path = req.originalFilePath.strip()

    validate_minio_object_exists(original_file_path)

    file_name = (
        req.fileName
        or get_file_name_from_minio_uri(original_file_path)
    )

    cr_body = build_cr_body(
        check_name=check_name,
        file_id=str(req.fileId),
        topic_id=int(req.topicId),
        subject_id=subject_id,
        file_name=file_name,
        original_file_path=original_file_path,
    )

    create_plagiarism_cr(cr_body)

    return {
        "checkName": check_name,
        "status": "queued",
        "file_id": str(req.fileId),
        "topic_id": int(req.topicId),
        "subject_id": subject_id,
        "file_name": file_name,
        "originalFilePath": original_file_path,
        "resultUrl": f"/plagiarism-checks/{check_name}/result",
    }


@router.get("/{check_name}/result")
async def get_result(check_name: str):
    record = await get_check_record(check_name)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Check not found: {check_name}",
        )

    base_response = {
        "checkName": record["check_name"],
        "file_id": record["file_id"],
        "topic_id": record["topic_id"],
        "subject_id": record["subject_id"],
        "file_name": record["file_name"],
        "original_file_path": record["input_pdf_path"],
        "candidate_count": record["candidate_count"],
        "total_sentences": record["total_sentences"],
        "plagiarized_sentences": record["plagiarized_sentences"],
        "plagiarism_ratio": (
            float(record["plagiarism_ratio"])
            if record["plagiarism_ratio"] is not None
            else None
        ),
        "is_plagiarized": record["is_plagiarized"],
        "created_at": (
            record["created_at"].isoformat()
            if record["created_at"]
            else None
        ),
        "started_at": (
            record["started_at"].isoformat()
            if record["started_at"]
            else None
        ),
        "completed_at": (
            record["completed_at"].isoformat()
            if record["completed_at"]
            else None
        ),
        "updated_at": (
            record["updated_at"].isoformat()
            if record["updated_at"]
            else None
        ),
    }

    status = record["status"]

    if status == "completed":
        return {
            **base_response,
            "status": "completed",
            "result": parse_result(record["result"]),
            "error": None,
        }

    if status == "failed":
        return {
            **base_response,
            "status": "failed",
            "result": None,
            "error": record["error_message"] or "Unknown error",
        }

    if status in {"queued", "processing"}:
        return {
            **base_response,
            "status": status,
            "result": None,
            "error": None,
        }

    return {
        **base_response,
        "status": status,
        "result": parse_result(record["result"]),
        "error": record["error_message"],
    }