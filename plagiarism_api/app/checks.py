import json
import os
import re
from typing import Any
from uuid import uuid4

import redis
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from minio import Minio
from minio.error import S3Error


router = APIRouter(
    prefix="/plagiarism-checks",
    tags=["Plagiarism Checks"],
)


# ===== Kubernetes config =====

K8S_GROUP = os.getenv("PLAGIARISM_CRD_GROUP", "plagiarism.io")
K8S_VERSION = os.getenv("PLAGIARISM_CRD_VERSION", "v1")
K8S_PLURAL = os.getenv("PLAGIARISM_CRD_PLURAL", "plagiarismchecks")
K8S_NAMESPACE = os.getenv("PLAGIARISM_NAMESPACE", "plagiarism")

API_INTERNAL_BASE_URL = os.getenv(
    "API_INTERNAL_BASE_URL",
    "http://plagiarism-api-svc.plagiarism.svc.cluster.local:8000",
)


# ===== MinIO config =====

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio-svc-private.storage.svc.cluster.local:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

UPLOAD_BUCKET = os.getenv("MINIO_UPLOAD_BUCKET", "uploads")


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


def sanitize_file_name(file_name: str) -> str:
    file_name = file_name.strip()
    file_name = file_name.replace("\\", "/").split("/")[-1]
    file_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)

    if not file_name:
        file_name = "document.pdf"

    if not file_name.lower().endswith(".pdf"):
        file_name = f"{file_name}.pdf"

    return file_name


def ensure_bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


def upload_pdf_to_minio(file: UploadFile, check_name: str) -> str:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. content_type={file.content_type}",
        )

    file_name = sanitize_file_name(file.filename or "document.pdf")
    object_name = f"{check_name}/{file_name}"

    minio_client = get_minio_client()

    try:
        ensure_bucket_exists(minio_client, UPLOAD_BUCKET)

        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        minio_client.put_object(
            bucket_name=UPLOAD_BUCKET,
            object_name=object_name,
            data=file.file,
            length=file_size,
            content_type=file.content_type,
        )

    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot upload file to MinIO: {e}",
        )

    return f"minio://{UPLOAD_BUCKET}/{object_name}"


def build_cr_body(
    *,
    check_name: str,
    subject_id: str,
    original_file_path: str,
    compare_pods: 5,
) -> dict[str, Any]:
    return {
        "apiVersion": f"{K8S_GROUP}/{K8S_VERSION}",
        "kind": "plagiarismcheck",
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
            detail=f"Cannot create PlagiarismCheck CR: {e.reason}",
        )


@router.post("/upload")
async def upload_and_submit(
    fileId: str = Form(...),
    topicId: int = Form(...),
    subjectId: str = Form(...),
    fileName: str = Form(...),
    comparePods: int = Form(5),
    file: UploadFile = File(...),
    checkName: str | None = Form(None),
):
    if comparePods <= 0:
        raise HTTPException(
            status_code=400,
            detail="comparePods must be greater than 0",
        )

    check_name = normalize_check_name(checkName)
    subject_id = str(subjectId)

    r = get_redis_client()

    r.setex(status_key(check_name), RESULT_TTL_SECONDS, "submitted")

    original_file_path = upload_pdf_to_minio(
        file=file,
        check_name=check_name,
    )

    

    cr_body = build_cr_body(
        check_name=check_name,
        subject_id=subject_id,
        original_file_path=original_file_path,
        compare_pods=comparePods,
        
    )

    create_plagiarism_cr(cr_body)

    metadata = {
        "file_id": str(fileId),
        "topic_id": int(topicId),
        "subject_id": str(subjectId),
        "file_name": fileName,
    }

    r.setex(
        metadata_key(check_name),
        RESULT_TTL_SECONDS,
        json.dumps(metadata, ensure_ascii=False),
    )

    return {
        "checkName": check_name,
        "status": "submitted",
        "file_id": str(fileId),
        "topic_id": int(topicId),
        "subject_id": subject_id,
        "file_name": fileName,
        "originalFilePath": original_file_path,
        "comparePods": comparePods,
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

