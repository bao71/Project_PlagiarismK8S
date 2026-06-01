from plagiarism.app.models.check import CheckResponse, MatchedSentence, ReferenceMatch
from plagiarism.app.repositories import milvus_repo
from plagiarism.app.services.preprocessing import SentenceRecord

SENTENCE_SIMILARITY_THRESHOLD = 0.8   # câu giống >80% → đạo văn
PLAGIARISM_CONCLUSION_THRESHOLD = 0.8  # P >80% → kết luận đạo văn


def check_against_reference(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    candidate: dict,                    # {document_id, file_name, subject_id, jaccard_similarity}
    sentence_labels: list[int],         # mảng nhãn 0/1, được cập nhật in-place
) -> ReferenceMatch:
    """
    So sánh chi tiết từng câu của tài liệu đẩy lên (d)
    với tài liệu tham chiếu (d1/d2/...).

    Cập nhật sentence_labels in-place.
    Trả về ReferenceMatch.
    """
    document_id = candidate["document_id"]

    # Query Milvus: với mỗi câu của d, tìm câu giống nhất trong d1
    match_results = milvus_repo.search_similar_sentences(
        query_embeddings=query_embeddings,
        document_id=document_id,
        top_k=1,
        similarity_threshold=SENTENCE_SIMILARITY_THRESHOLD,
    )

    matched_sentences: list[MatchedSentence] = []
    plagiarized_count = 0

    for i, (sent, match) in enumerate(zip(query_sentences, match_results)):
        if match is not None:
            # Câu này bị đạo văn từ tài liệu tham chiếu
            sentence_labels[i] = 1
            plagiarized_count += 1
            matched_sentences.append(MatchedSentence(
                query_sentence_index=sent.sentence_index,
                query_sentence_text=sent.sentence_text,
                query_page=sent.page_number,
                ref_sentence_text=match["sentence_text"],
                ref_page=match["page_number"],
                similarity=match["similarity"],
            ))

    total = len(query_sentences)
    plagiarism_ratio = round(plagiarized_count / total, 4) if total > 0 else 0.0

    return ReferenceMatch(
        document_id=document_id,
        file_name=candidate["file_name"],
        subject_id=candidate["subject_id"],
        jaccard_similarity=candidate["jaccard_similarity"],
        plagiarism_ratio=plagiarism_ratio,
        plagiarized_count=plagiarized_count,
        matched_sentences=matched_sentences,
    )


def run_plagiarism_check(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    candidates: list[dict],
) -> CheckResponse:
    """
    Chạy toàn bộ luồng so khớp chi tiết.

    candidates: danh sách tài liệu tham chiếu đã qua lọc thô MinHash.
    """
    total_sentences = len(query_sentences)
    sentence_labels = [0] * total_sentences   # 0 = chưa đạo văn, 1 = đạo văn
    references: list[ReferenceMatch] = []

    for candidate in candidates:
        ref_match = check_against_reference(
            query_sentences=query_sentences,
            query_embeddings=query_embeddings,
            candidate=candidate,
            sentence_labels=sentence_labels,
        )
        # Chỉ đưa vào kết quả nếu có ít nhất 1 câu trùng
        if ref_match.plagiarized_count > 0:
            references.append(ref_match)

    plagiarized_sentences = sum(sentence_labels)
    plagiarism_ratio = round(plagiarized_sentences / total_sentences, 4) if total_sentences > 0 else 0.0
    is_plagiarized = plagiarism_ratio >= PLAGIARISM_CONCLUSION_THRESHOLD

    return CheckResponse(
        total_sentences=total_sentences,
        plagiarized_sentences=plagiarized_sentences,
        plagiarism_ratio=plagiarism_ratio,
        is_plagiarized=is_plagiarized,
        sentence_labels=sentence_labels,
        references=sorted(references, key=lambda x: x.plagiarism_ratio, reverse=True),
    )
