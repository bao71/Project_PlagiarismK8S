from app.models.check import CheckResponse, MatchedSentence, ReferenceMatch
from app.repositories import milvus_repo
from app.services.preprocessing import SentenceRecord
from pymilvus import Collection
import json
SENTENCE_SIMILARITY_THRESHOLD = 0.8
PLAGIARISM_CONCLUSION_THRESHOLD = 0.8


def redis_matched_key(check_name: str) -> str:
    return f"plagiarism:{check_name}:matched_sentences"


def redis_source_key(check_name: str) -> str:
    return f"plagiarism:{check_name}:sentence_sources"

def check_against_single_reference(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    active_indices: list[int],
    candidate: dict,
    sentence_labels: list[int],
    collection: Collection,
) -> tuple[int, list[MatchedSentence], float]:

    document_id = candidate["document_id"]
    matched_sentences = []

    active_embeddings = [query_embeddings[c] for c in active_indices]
    if not active_embeddings:
        return 0, [], 0.0

    results = collection.search(
        data=active_embeddings,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=1,
        expr=f"document_id == '{document_id}'",
        output_fields=["sentence_text", "page_number"],
    )

    total_ref_sentences = 1
    plagiarized_count = 0

    for c, hits in zip(active_indices, results):
        if not hits:
            continue
        best = hits[0]
        sim = best.score
        if sim < SENTENCE_SIMILARITY_THRESHOLD:
            continue
        sentence_labels[c] = 1
        plagiarized_count += 1
        matched_sentences.append(MatchedSentence(
            query_sentence_index=c,
            query_sentence_text=query_sentences[c].sentence_text,
            query_page=query_sentences[c].page_number,
            ref_sentence_text=best.entity.get("sentence_text"),
            ref_page=best.entity.get("page_number"),
            similarity=float(sim),
        ))

    if plagiarized_count > 0:
        res = collection.query(expr=f"document_id == '{document_id}'", output_fields=["count(*)"])
        total_ref_sentences = res[0].get("count(*)", 1) if res else 1
        plagiarism_ratio = round(plagiarized_count / total_ref_sentences, 4)
    else:
        plagiarism_ratio = 0.0

    return plagiarized_count, matched_sentences, plagiarism_ratio


def check(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    candidates: list[dict],
    sentence_labels: list[int],
    redis_client=None,
    check_name: str | None = None,
) -> list[ReferenceMatch]:
    milvus_repo.connect_milvus()

    collection = Collection("PlagiarismDetection")
    collection.load()

    reference_matches: list[ReferenceMatch] = []

    matched_key = None
    source_key = None

    if redis_client is not None and check_name:
        matched_key = redis_matched_key(check_name)
        source_key = redis_source_key(check_name)

    for candidate in candidates:
        redis_matched_sentences = set()

        # Lấy toàn bộ sentence_index đã match từ Redis một lần.
        # Yêu cầu redis_client tạo với decode_responses=True.
        if redis_client is not None and matched_key:
            try:
                redis_matched_sentences = redis_client.smembers(matched_key)
            except Exception as e:
                print(
                    f"Redis read matched_sentences error: {e}",
                    flush=True,
                )
                redis_matched_sentences = set()

        active_indices = []
        redis_skipped = 0

        # Chỉ kiểm tra Redis, không kiểm tra local lbl == 1.
        for c in range(len(sentence_labels)):
            if str(c) in redis_matched_sentences:
                sentence_labels[c] = 1
                redis_skipped += 1
                continue

            active_indices.append(c)

        print(
            f"candidate={candidate.get('document_id')} "
            f"active_indices={len(active_indices)} "
            f"redis_skipped={redis_skipped}",
            flush=True,
        )

        if not active_indices:
            break

        try:
            p_count, matches, p_ratio = check_against_single_reference(
                query_sentences=query_sentences,
                query_embeddings=query_embeddings,
                active_indices=active_indices,
                candidate=candidate,
                sentence_labels=sentence_labels,
                collection=collection,
            )

            if p_count > 0:
                # Ghi các câu vừa match vào Redis để Pod khác skip ở candidate sau.
                if redis_client is not None and matched_key:
                    try:
                        pipe = redis_client.pipeline()

                        for match in matches:
                            sentence_index = match.query_sentence_index

                            pipe.sadd(
                                matched_key,
                                str(sentence_index),
                            )

                            if source_key:
                                pipe.hset(
                                    source_key,
                                    str(sentence_index),
                                    json.dumps(
                                        {
                                            "sentence_index": sentence_index,
                                            "document_id": str(candidate["document_id"]),
                                            "file_name": candidate["file_name"],
                                            "subject_id": candidate["subject_id"],
                                            "group_id": candidate.get("group_id"),
                                            "jaccard_similarity": candidate.get(
                                                "jaccard_similarity"
                                            ),
                                            "similarity": match.similarity,
                                            "ref_page": match.ref_page,
                                            "query_page": match.query_page,
                                        },
                                        ensure_ascii=False,
                                    ),
                                )

                        pipe.execute()

                    except Exception as e:
                        print(
                            f"Redis write matched_sentences error: {e}",
                            flush=True,
                        )

                reference_matches.append(
                    ReferenceMatch(
                        document_id=str(candidate["document_id"]),
                        file_name=candidate["file_name"],
                        subject_id=candidate["subject_id"],
                        group_id=candidate.get("group_id"),
                        jaccard_similarity=candidate["jaccard_similarity"],
                        plagiarism_ratio=p_ratio,
                        plagiarized_count=p_count,
                        matched_sentences=matches,
                    )
                )

        except Exception as e:
            print(
                f"Error checking document {candidate['document_id']}: {e}",
                flush=True,
            )

    return reference_matches

def run_plagiarism_check(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    candidates: list[dict],
    redis_client=None,
    check_name: str | None = None,
) -> CheckResponse:
    total_sentences = len(query_sentences)
    sentence_labels = [0] * total_sentences

    ref_match = check(
        query_sentences=query_sentences,
        query_embeddings=query_embeddings,
        candidates=candidates,
        sentence_labels=sentence_labels,
        redis_client=redis_client,
        check_name=check_name,
    )

    total_plagiarized = sum(sentence_labels)

    plagiarism_ratio = (
        round(total_plagiarized / total_sentences, 4)
        if total_sentences > 0
        else 0.0
    )

    return CheckResponse(
        total_sentences=total_sentences,
        plagiarized_sentences=total_plagiarized,
        plagiarism_ratio=plagiarism_ratio,
        is_plagiarized=plagiarism_ratio > PLAGIARISM_CONCLUSION_THRESHOLD,
        sentence_labels=sentence_labels,
        references=ref_match,
    )


def find_candidate(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    candidates: list[dict],
) -> CheckResponse:
    if not candidates:
        return CheckResponse(
            total_sentences=len(query_sentences),
            plagiarized_sentences=0,
            plagiarism_ratio=0.0,
            is_plagiarized=False,
            sentence_labels=[0] * len(query_sentences),
            references=[],
        )
    
    best = max(candidates, key=lambda c: c["jaccard_similarity"])

    result = run_plagiarism_check(
        query_sentences=query_sentences,
        query_embeddings=query_embeddings,
        candidates=[best],
    )
    return result