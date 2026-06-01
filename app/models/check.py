from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MatchedSentence(BaseModel):
    query_sentence_index: int
    query_sentence_text: str
    query_page: int
    ref_sentence_text: str
    ref_page: int
    similarity: float


class ReferenceMatch(BaseModel):
    document_id: str
    file_name: str
    subject_id: str                  # giữ nguyên, dùng nội bộ (không expose trong report)
    group_id: Optional[int] = None   # group_id nếu có, mặc định None
    jaccard_similarity: float        # minhash similarity (lọc thô)
    plagiarism_ratio: float          # số câu đạo văn từ tài liệu này / tổng câu ref
    plagiarized_count: int           # số câu bị đạo văn từ tài liệu này
    matched_sentences: list[MatchedSentence]


class CheckResponse(BaseModel):
    total_sentences: int             # tổng số câu của tài liệu đẩy lên
    plagiarized_sentences: int       # số câu bị gán nhãn đạo văn
    plagiarism_ratio: float          # P = số câu đạo văn / tổng câu
    is_plagiarized: bool             # P > 80%
    sentence_labels: list[int] = []       # 0/1 cho từng câu của d
    references: list[ReferenceMatch]


class CheckReport(BaseModel):
    submission_id: Optional[int] = None
    topic_id: Optional[int] = None
    file_name: str
    plagiarism_score: float       
    status: str = "checked"
    checked_at: datetime
    report: CheckResponse