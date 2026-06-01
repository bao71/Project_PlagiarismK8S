import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model: SentenceTransformer | None = None


def load_model():
    """Load model 1 lần khi app khởi động"""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_sentences(sentences: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed danh sách câu.
    Trả về list[list[float]] — mỗi phần tử là vector 768 chiều.
    """
    if not sentences:
        return []

    model = load_model()
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    result = embeddings.tolist()

    # Validate từng vector
    for i, vec in enumerate(result):
        if len(vec) != 768:
            raise ValueError(f"Câu {i} có dim={len(vec)}, expected 768. Nội dung: '{sentences[i][:50]}'")

    return result