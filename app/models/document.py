from pydantic import BaseModel
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    file_name: str
    subject_id: str
    file_path: str
    sentences_indexed: int


class UploadBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    documents: list[DocumentUploadResponse]
    errors: list[dict]
