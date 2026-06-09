from fastapi import FastAPI

from app.checks import router as checks_router


app = FastAPI(
    title="Plagiarism Check API",
    version="1.0.0",
)

app.include_router(checks_router)


@app.get("/health")
def health():
    return {"status": "ok"}