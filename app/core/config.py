from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # PostgreSQL dùng cho MinHash candidate filtering
    minhash_postgres_dsn: str

    # PostgreSQL dùng để lưu kết quả cuối
    result_postgres_dsn: str

    # MinIO
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False

    # Milvus
    milvus_host: str
    milvus_port: int
    milvus_collection_name: str = "PlagiarismDetection"
    milvus_replica_number: int = 1

    # Embedding
    embedding_model: str

    # MinHash
    minhash_threshold: float = 0.03

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()