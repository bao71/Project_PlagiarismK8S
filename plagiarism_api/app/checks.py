import json
import os
import re
from typing import Any
from uuid import uuid4

import redis
from fastapi import APIRouter, Form, HTTPException
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from minio import Minio
from minio.error import S3Error


router = APIRouter(
    prefix="/plagiarism-checks",
    tags=["Plagiarism Checks"],
)

from pydantic import BaseModel


class SubmitCheckRequest(BaseModel):
    fileId: str
    topicId: int
    subjectId: str
    originalFilePath: str
    fileName: str | None = None
    checkName: str | None = None

# ===== Kubernetes config =====

K8S_GROUP = os.getenv("PLAGIARISM_CRD_GROUP", "plagiarism.io")
K8S_VERSION = os.getenv("PLAGIARISM_CRD_VERSION", "v1")
K8S_PLURAL = os.getenv("PLAGIARISM_CRD_PLURAL", "plagiarismchecks")
K8S_NAMESPACE = os.getenv("PLAGIARISM_NAMESPACE", "plagiarism")
K8S_KIND = os.getenv("PLAGIARISM_CRD_KIND", "plagiarismcheck")


# ===== MinIO config =====

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Có validate file tồn tại trên MinIO hay không
VALIDATE_MINIO_OBJECT = os.getenv("VALIDATE_MINIO_OBJECT", "true").lower() == "true"


# ===== Redis config =====

REDIS_HOST = os.getenv("REDIS_HOST", "redis.cache.svc.cluster.local")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "86400"))


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


def metadata_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:metadata"


def status_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:status"


def result_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:result"


def error_key(check_name: str) -> str:
    return f"plagiarism:check:{check_name}:error"


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

ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx"}


def is_allowed_document(object_name: str) -> bool:
    object_name = object_name.lower()
    return any(object_name.endswith(ext) for ext in ALLOWED_FILE_EXTENSIONS)


def parse_minio_uri(minio_uri: str) -> tuple[str, str]:
    """
    Input:
        minio://uploads/check-001/case1.pdf
        minio://uploads/check-001/report.docx

    Output:
        bucket = uploads
        object_name = check-001/case1.pdf
    """
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

    path = minio_uri.replace("minio://", "", 1)

    parts = path.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid MinIO path. Expected format: minio://bucket/object",
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
    return object_name.split("/")[-1]


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
    except S3Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot access MinIO object: {minio_uri}. Error: {e}",
        )


def build_cr_body(
    *,
    check_name: str,
    subject_id: str,
    original_file_path: str,
    compare_pods: int,
) -> dict[str, Any]:
    return {
        "apiVersion": f"{K8S_GROUP}/{K8S_VERSION}",
        "kind": K8S_KIND,
        "metadata": {
            "name": check_name,
            "namespace": K8S_NAMESPACE,
        },
        "spec": {
            "subjectId": subject_id,
            "originalFilePath": original_file_path,
            "comparePods": compare_pods,
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

    except ApiException as e:
        if e.status == 409:
            raise HTTPException(
                status_code=409,
                detail=f"Check already exists: {cr_body['metadata']['name']}",
            )

        raise HTTPException(
            status_code=500,
            detail=f"Cannot create PlagiarismCheck CR: {e.reason}. Body: {e.body}",
        )


@router.post("/submit")
async def submit_from_minio(req: SubmitCheckRequest):
    check_name = normalize_check_name(req.checkName)
    subject_id = str(req.subjectId)
    original_file_path = req.originalFilePath.strip()

    validate_minio_object_exists(original_file_path)

    file_name = req.fileName
    if not file_name:
        file_name = get_file_name_from_minio_uri(original_file_path)

    cr_body = build_cr_body(
        check_name=check_name,
        subject_id=subject_id,
        original_file_path=original_file_path,
        compare_pods=5,
    )

    create_plagiarism_cr(cr_body)

    metadata = {
        "file_id": str(req.fileId),
        "topic_id": int(req.topicId),
        "subject_id": str(req.subjectId),
        "file_name": file_name,
        "original_file_path": original_file_path,
    }

    r = get_redis_client()

    r.setex(status_key(check_name), RESULT_TTL_SECONDS, "submitted")

    r.setex(
        metadata_key(check_name),
        RESULT_TTL_SECONDS,
        json.dumps(metadata, ensure_ascii=False),
    )

    return {
        "checkName": check_name,
        "status": "submitted",
        "file_id": str(req.fileId),
        "topic_id": int(req.topicId),
        "subject_id": subject_id,
        "file_name": file_name,
        "originalFilePath": original_file_path,
        "resultUrl": f"/plagiarism-checks/{check_name}/result",
    }

@router.get("/{check_name}/result")
def get_result(check_name: str):
    r = get_redis_client()

    status = r.get(status_key(check_name))
    raw_result = r.get(result_key(check_name))
    error = r.get(error_key(check_name))

    raw_metadata = r.get(metadata_key(check_name))
    metadata = json.loads(raw_metadata) if raw_metadata else {}

    base_response = {
        "checkName": check_name,
        "file_id": metadata.get("file_id"),
        "topic_id": metadata.get("topic_id"),
        "subject_id": metadata.get("subject_id"),
        "file_name": metadata.get("file_name"),
        "original_file_path": metadata.get("original_file_path"),
    }

    if raw_result:
        return {
            **base_response,
            "status": "completed",
            "result": json.loads(raw_result),
        }

    if status == "failed":
        return {
            **base_response,
            "status": "failed",
            "error": error or "Unknown error",
            "result": None,
        }

    if status in {"submitted", "running"}:
        return {
            **base_response,
            "status": status,
            "result": None,
        }

    raise HTTPException(
        status_code=404,
        detail=f"Check not found: {check_name}",
    )