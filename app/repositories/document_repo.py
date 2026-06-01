import uuid
import asyncpg
from app.services.minhash import jaccard_similarity


async def insert_document(
    conn: asyncpg.Connection,
    file_name: str,
    subject_id: str,
    file_path: str,
) -> str:
    doc_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO documents (id, file_name, subject_id, file_path, minhash, created_at)
        VALUES ($1, $2, $3, $4, NULL, NOW())
        """,
        doc_id, file_name, subject_id, file_path,
    )
    return doc_id


async def update_minhash(
    conn: asyncpg.Connection,
    doc_id: str,
    minhash: list[int],
):
    await conn.execute(
        "UPDATE documents SET minhash = $1 WHERE id = $2",
        minhash, doc_id,
    )


async def get_document_by_id(
    conn: asyncpg.Connection,
    doc_id: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        "SELECT * FROM documents WHERE id = $1", doc_id
    )


async def document_exists(
    conn: asyncpg.Connection,
    file_name: str,
    subject_id: str,
) -> bool:
    row = await conn.fetchrow(
        "SELECT id FROM documents WHERE file_name = $1 AND subject_id = $2",
        file_name, subject_id,
    )
    return row is not None