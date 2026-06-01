import plagiarism.app.patch

from contextlib import asynccontextmanager

from fastapi import FastAPI

from plagiarism.app.api.v1.router import router as api_v1_router
from plagiarism.app.core.database import close_db, init_db
from plagiarism.app.repositories.milvus_repo import create_collection_if_not_exists
from plagiarism.app.services.embedding import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Kết nối Postgres...")
    await init_db()
    print("Khởi tạo Milvus collection...")
    create_collection_if_not_exists()
    print("Load embedding model...")
    load_model()
    print("App sẵn sàng")
    yield
    print("Đóng kết nối Postgres...")
    await close_db()


app = FastAPI(
    title="Plagiarism Detection API",
    description="API chuẩn bị tài liệu tham chiếu cho hệ thống kiểm tra đạo văn",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}