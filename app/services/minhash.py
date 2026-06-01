from datasketch import MinHash
import asyncpg


NUM_PERM = 128
SHINGLE_SIZE = 3


def compute_minhash(text: str) -> list[int]:
    """
    Tính MinHash từ text đã clean.
    Trả về mảng NUM_PERM số nguyên để lưu vào Postgres INTEGER[].
    """

    m = MinHash(num_perm=NUM_PERM)
    words = text.split()

    if len(words) < SHINGLE_SIZE:
        for word in words:
            m.update(word.encode("utf-8"))
    else:
        for i in range(len(words) - SHINGLE_SIZE + 1):
            shingle = " ".join(words[i:i + SHINGLE_SIZE])
            m.update(shingle.encode("utf-8"))

    return [int(v) for v in m.hashvalues]


def jaccard_similarity(
    minhash_a: list[int],
    minhash_b: list[int],
) -> float:
    """
    Ước tính Jaccard similarity giữa 2 MinHash vector.
    """

    if len(minhash_a) != len(minhash_b):
        raise ValueError(
            "Hai MinHash phải có cùng số lượng permutations"
        )

    matches = sum(
        a == b
        for a, b in zip(minhash_a, minhash_b)
    )

    return matches / NUM_PERM


async def find_candidates_by_minhash(
    conn: asyncpg.Connection,
    minhash: list[int],
    subject_id: str,
    threshold: float,
) -> list[dict]:
    """
    Lọc thô:
    - lấy các tài liệu tham chiếu cùng subject_id
    - tính Jaccard similarity qua MinHash
    - trả về candidates có similarity >= threshold
    """

    rows = await conn.fetch(
        """
        SELECT
            id,
            file_name,
            subject_id,
            minhash
        FROM documents
        WHERE subject_id = $1
          AND minhash IS NOT NULL
        """,
        subject_id,
    )

    candidates: list[dict] = []

    for row in rows:
        ref_minhash = list(row["minhash"])

        sim = jaccard_similarity(
            minhash,
            ref_minhash,
        )

        if sim >= threshold:
            candidates.append({
                "document_id": str(row["id"]),
                "file_name": row["file_name"],
                "subject_id": row["subject_id"],
                "group_id": None,
                "jaccard_similarity": round(sim, 4),
            })

    return sorted(
        candidates,
        key=lambda x: x["jaccard_similarity"],
        reverse=True,
    )