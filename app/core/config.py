from pydantic_settings import BaseSettings, SettingsConfigDict
from base64 import b64decode


class Settings(BaseSettings):
    # Postgres
    postgres_dsn: str

    # MinIO
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str

    # Milvus
    milvus_host: str
    milvus_port: int

    # Embedding
    embedding_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
