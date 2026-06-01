import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from plagiarism.app.core.config import settings


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_file(file_bytes: bytes, file_path: str, content_type: str = "application/pdf") -> str:
    """
    Upload file lên MinIO.
    file_path: đường dẫn trong bucket, ví dụ 'documents/<uuid>/filename.pdf'
    Trả về file_path để lưu vào Postgres.
    """
    client = _get_client()
    client.put_object(
        Bucket=settings.minio_bucket,
        Key=file_path,
        Body=file_bytes,
        ContentType=content_type,
    )
    return file_path


def download_file(file_path: str) -> bytes:
    """Tải file từ MinIO về dạng bytes"""
    client = _get_client()
    response = client.get_object(
        Bucket=settings.minio_bucket,
        Key=file_path,
    )
    return response["Body"].read()


def file_exists(file_path: str) -> bool:
    client = _get_client()
    try:
        client.head_object(Bucket=settings.minio_bucket, Key=file_path)
        return True
    except ClientError:
        return False
