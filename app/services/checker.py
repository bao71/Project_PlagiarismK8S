from app.models.check import CheckResponse, MatchedSentence, ReferenceMatch
from app.repositories import milvus_repo
from app.services.preprocessing import SentenceRecord
from pymilvus import Collection
from underthesea import word_tokenize
import os

PLAGIARISM_CONCLUSION_THRESHOLD = 0.8
LEXICAL_NGRAM = 2
ALPHA = 0.5
FINAL_SCORE_THRESHOLD = 0.5
ANISOTROPY_LEX_MAX = 0.05
ANISOTROPY_SEM_MIN = 0.95

_PUNCT_TOKENS = {".", "!", "?", ",", ";", ":"}

collection_name = os.getenv("MILVUS_COLLECTION_NAME", "PlagiarismDetection")


def get_milvus_replica_number() -> int:
    value = os.getenv("MILVUS_REPLICA_NUMBER")

    if value is None:
        raise RuntimeError(
            "Missing env MILVUS_REPLICA_NUMBER. "
            "Please set MILVUS_REPLICA_NUMBER=1 or 2 in Kubernetes Pod/Job env."
        )

    return int(value)

def segment_for_lexical(text: str, cache: dict[str, str]) -> str:
    if text not in cache:
        cache[text] = word_tokenize(text, format="text")
    return cache[text]

def get_ngrams(text: str, cache: dict[str, str], n: int = LEXICAL_NGRAM) -> set:
    segmented = segment_for_lexical(text, cache)
    tokens = [t for t in segmented.split() if t not in _PUNCT_TOKENS]
    if len(tokens) < n:
        return {tuple(tokens)}
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def lexical_score(query_text: str, ref_text: str, cache: dict[str, str]) -> float:
    q_grams = get_ngrams(query_text, cache)
    r_grams = get_ngrams(ref_text, cache)
    union = q_grams | r_grams
    if not union:
        return 0.0
    return len(q_grams & r_grams) / len(union)

def combine_scores(lexical_score: float, semantic_score: float) -> dict:
    final_score = ALPHA * lexical_score + (1 - ALPHA) * semantic_score
    is_anisotropy_flag = (
        lexical_score < ANISOTROPY_LEX_MAX and semantic_score > ANISOTROPY_SEM_MIN
    )
    return {
        "final_score": round(float(final_score), 4),
        "is_anisotropy_flag": is_anisotropy_flag,
        "is_similar": final_score >= FINAL_SCORE_THRESHOLD and not is_anisotropy_flag,
    }

def check_against_single_reference(
    query_sentences: list[SentenceRecord],
    query_embeddings: list[list[float]],
    active_indices: list[int],
    candidate: dict,
    sentence_labels: list[int],
    collection: Collection,
    segment_cache: dict[str, str],
) -> tuple[int, list[MatchedSentence], float]:

    document_id = candidate["document_id"]
    matched_sentences = []

    active_embeddings = [query_embeddings[c] for c in active_indices]
    if not active_embeddings:
        return 0, [], 0.0

    results = collection.search(
        data=active_embeddings,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"efsearch": 64}},
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
        sim = float(best.score)
        ref_text = best.entity.get("sentence_text")
        query_text = query_sentences[c].sentence_text

        lex_score = lexical_score(query_text, ref_text, segment_cache)
        combined = combine_scores(lex_score, sim)

        if not combined["is_similar"]:
            continue

        sentence_labels[c] = 1
        plagiarized_count += 1
        matched_sentences.append(MatchedSentence(
            query_sentence_index=c,
            query_sentence_text=query_text,
            query_page=query_sentences[c].page_number,
            ref_sentence_text=ref_text,
            ref_page=best.entity.get("page_number"),
            similarity=sim,
            lexical_similarity=round(lex_score, 4),
            final_score=combined["final_score"],
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
    segment_cache: dict[str, str],
    check_name: str | None = None,
) -> list[ReferenceMatch]:
    milvus_repo.connect_milvus()

    replica_number = get_milvus_replica_number()
    collection = Collection(collection_name)

    print(
        f"Before Milvus load: "
        f"collection={collection_name}, "
        f"replica_number={replica_number}, "
        f"MILVUS_REPLICA_NUMBER_ENV={os.getenv('MILVUS_REPLICA_NUMBER')}, "
        f"MILVUS_HOST={os.getenv('MILVUS_HOST')}, "
        f"CHECK_NAME={check_name}",
        flush=True,
    )

    collection.load(replica_number=replica_number)

    print(
        f"Loaded Milvus collection={collection_name} "
        f"replica_number={replica_number}",
        flush=True,
    )

    reference_matches: list[ReferenceMatch] = []

    for candidate in candidates:

        active_indices = []

        for local_index, label in enumerate(
            sentence_labels
        ):
            if label == 1:
                continue

            active_indices.append(local_index)

        print(
            f"candidate={candidate.get('document_id')} "
            f"active_indices={len(active_indices)}",
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
                segment_cache=segment_cache,
            )

            if p_count > 0:
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
    check_name: str | None = None,
) -> CheckResponse:
    total_sentences = len(query_sentences)
    sentence_labels = [0] * total_sentences
    segment_cache: dict[str, str] = {}

    ref_match = check(
        query_sentences=query_sentences,
        query_embeddings=query_embeddings,
        candidates=candidates,
        sentence_labels=sentence_labels,
        segment_cache=segment_cache,
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
        sentence_labels=[],
        references=ref_match,
    )