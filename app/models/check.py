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
    lexical_similarity: float
    final_score: float


class ReferenceMatch(BaseModel):
    document_id: str
    file_name: str
    subject_id: str
    group_id: Optional[int] = None
    jaccard_similarity: float
    plagiarism_ratio: float
    plagiarized_count: int
    matched_sentences: list[MatchedSentence]


class CheckResponse(BaseModel):
    total_sentences: int
    plagiarized_sentences: int
    plagiarism_ratio: float
    is_plagiarized: bool
    sentence_labels: list[int] = []
    references: list[ReferenceMatch]


class CheckReport(BaseModel):
    submission_id: Optional[int] = None
    topic_id: Optional[int] = None
    file_name: str
    plagiarism_score: float       
    status: str = "checked"
    checked_at: datetime
    report: CheckResponse