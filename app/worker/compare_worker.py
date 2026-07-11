import os
import time
from typing import Any, Dict

import requests

from app.services.checker import run_plagiarism_check
from app.services.preprocessing import SentenceRecord


def convert_check_response_to_dict(
    check_response: Any,
) -> Dict[str, Any]:
    if hasattr(check_response, "model_dump"):
        return check_response.model_dump()

    if hasattr(check_response, "dict"):
        return check_response.dict()

    if isinstance(check_response, dict):
        return check_response

    raise TypeError(
        f"Unsupported response type: {type(check_response)}"
    )


def convert_sentences_to_records(
    sentences: list[Any],
    sentence_indices: list[int],
) -> list[SentenceRecord]:
    """
    Chuyển sentence dictionary thành SentenceRecord.

    sentence_indices được dùng làm global sentence index.
    Không cần thêm một vòng lặp kiểm tra riêng.
    """
    if len(sentences) != len(sentence_indices):
        raise ValueError(
            "Sentence count mismatch: "
            f"sentences={len(sentences)}, "
            f"indices={len(sentence_indices)}"
        )

    sentence_records: list[SentenceRecord] = []

    for local_index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            raise ValueError(
                "Invalid sentence type at "
                f"local_index={local_index}: "
                f"type={type(sentence)}"
            )

        sentence_text = sentence.get("sentence_text")

        if not sentence_text:
            raise ValueError(
                "Missing sentence_text at "
                f"local_index={local_index}: {sentence}"
            )

        global_sentence_index = int(
            sentence_indices[local_index]
        )

        # Xác thực ngay trong vòng convert hiện có,
        # không phát sinh thêm một vòng lặp.
        payload_sentence_index = sentence.get(
            "sentence_index"
        )

        if (
            payload_sentence_index is not None
            and int(payload_sentence_index)
            != global_sentence_index
        ):
            raise ValueError(
                "Global sentence index mismatch: "
                f"local_index={local_index}, "
                f"sentence_index="
                f"{payload_sentence_index}, "
                f"expected="
                f"{global_sentence_index}"
            )

        normalized = {
            "page_number": int(
                sentence.get("page_number", 0)
            ),
            "sentence_index": (
                global_sentence_index
            ),
            "sentence_index_page": int(
                sentence.get(
                    "sentence_index_page",
                    global_sentence_index,
                )
            ),
            "sentence_text": sentence_text,
            "bbox_x0": float(
                sentence.get("bbox_x0", 0.0)
            ),
            "bbox_y0": float(
                sentence.get("bbox_y0", 0.0)
            ),
            "bbox_x1": float(
                sentence.get("bbox_x1", 0.0)
            ),
            "bbox_y1": float(
                sentence.get("bbox_y1", 0.0)
            ),
        }

        sentence_records.append(
            SentenceRecord(**normalized)
        )

    return sentence_records


def main():
    worker_start = time.perf_counter()

    check_name = os.getenv("CHECK_NAME")
    stage1_url = os.getenv("STAGE1_URL")
    part_index = int(
        os.getenv("PART_INDEX", "0")
    )
    part_count = int(
        os.getenv("PART_COUNT", "1")
    )

    if not stage1_url:
        raise ValueError("Missing STAGE1_URL")

    if part_count <= 0:
        raise ValueError(
            f"Invalid PART_COUNT={part_count}"
        )

    if part_index < 0 or part_index >= part_count:
        raise ValueError(
            "Invalid partition: "
            f"part_index={part_index}, "
            f"part_count={part_count}"
        )

    print(
        "=== Start compare worker ===",
        flush=True,
    )
    print(
        f"STAGE1_URL={stage1_url}",
        flush=True,
    )
    print(
        f"PART_INDEX={part_index}",
        flush=True,
    )
    print(
        f"PART_COUNT={part_count}",
        flush=True,
    )
    print(
        f"CHECK_NAME={check_name}",
        flush=True,
    )

    response = requests.get(
        f"{stage1_url}/part",
        params={
            "part_index": part_index,
            "part_count": part_count,
        },
        timeout=300,
    )

    response.raise_for_status()

    payload = response.json()

    raw_sentences = payload.get("sentences")
    candidates = payload.get("candidates")
    query_embeddings = payload.get(
        "query_embeddings"
    )
    raw_sentence_indices = payload.get(
        "sentence_indices"
    )
    global_total_sentences = payload.get(
        "total_sentences"
    )

    if not isinstance(raw_sentences, list):
        raise ValueError(
            "Missing or invalid sentences "
            "from stage1 payload"
        )

    if not isinstance(candidates, list):
        raise ValueError(
            "Missing or invalid candidates "
            "from stage1 payload"
        )

    if not isinstance(query_embeddings, list):
        raise ValueError(
            "Missing or invalid query_embeddings "
            "from stage1 payload"
        )

    if not isinstance(
        raw_sentence_indices,
        list,
    ):
        raise ValueError(
            "Missing or invalid sentence_indices "
            "from stage1 payload"
        )

    if global_total_sentences is None:
        raise ValueError(
            "Missing total_sentences "
            "from stage1 payload"
        )

    sentence_indices = [
        int(index)
        for index in raw_sentence_indices
    ]

    if len(sentence_indices) != len(
        raw_sentences
    ):
        raise ValueError(
            "Sentence indices mismatch: "
            f"indices={len(sentence_indices)}, "
            f"sentences={len(raw_sentences)}"
        )

    if len(query_embeddings) != len(
        raw_sentences
    ):
        raise ValueError(
            "Embedding count mismatch: "
            f"sentences={len(raw_sentences)}, "
            f"embeddings={len(query_embeddings)}"
        )

    payload_part_index = int(
        payload.get("part_index", part_index)
    )
    payload_part_count = int(
        payload.get("part_count", part_count)
    )

    if payload_part_index != part_index:
        raise ValueError(
            "Stage1 part_index mismatch: "
            f"requested={part_index}, "
            f"received={payload_part_index}"
        )

    if payload_part_count != part_count:
        raise ValueError(
            "Stage1 part_count mismatch: "
            f"requested={part_count}, "
            f"received={payload_part_count}"
        )

    sentence_records = (
        convert_sentences_to_records(
            sentences=raw_sentences,
            sentence_indices=sentence_indices,
        )
    )

    print(
        f"Loaded sentences={len(sentence_records)}, "
        f"embeddings={len(query_embeddings)}, "
        f"candidates={len(candidates)}, "
        f"global_total_sentences="
        f"{global_total_sentences}",
        flush=True,
    )

    # Trường hợp partition rỗng hoặc không có candidate:
    # không cần gọi Milvus.
    if not sentence_records or not candidates:
        result = {
            "total_sentences": len(
                sentence_records
            ),
            "plagiarized_sentences": 0,
            "plagiarism_ratio": 0.0,
            "is_plagiarized": False,
            "sentence_labels": (
                [0] * len(sentence_records)
            ),
            "references": [],
            "plagiarism_check_seconds": 0.0,
        }

    else:
        print(
            "Running plagiarism check...",
            flush=True,
        )

        check_start = time.perf_counter()

        check_response = run_plagiarism_check(
            query_sentences=sentence_records,
            query_embeddings=query_embeddings,
            candidates=candidates,
        )

        check_seconds = round(
            time.perf_counter() - check_start,
            4,
        )

        print(
            "PLAGIARISM_CHECK_SECONDS="
            f"{check_seconds}",
            flush=True,
        )

        result = (
            convert_check_response_to_dict(
                check_response
            )
        )

        result[
            "plagiarism_check_seconds"
        ] = check_seconds

    # Các metadata này phải được thêm cho cả hai
    # trường hợp: có hoặc không có candidate.
    result["sentence_indices"] = (
        sentence_indices
    )
    result["partition_sentence_count"] = (
        len(sentence_indices)
    )
    result["global_total_sentences"] = int(
        global_total_sentences
    )
    result["part_index"] = part_index
    result["part_count"] = part_count
    result["candidate_count"] = len(
        candidates
    )

    total_compare_worker_seconds = round(
        time.perf_counter() - worker_start,
        4,
    )

    result[
        "total_compare_worker_seconds"
    ] = total_compare_worker_seconds

    print(
        "TOTAL_COMPARE_WORKER_SECONDS="
        f"{total_compare_worker_seconds}",
        flush=True,
    )

    print(
        "Posting part result back to stage1...",
        flush=True,
    )

    post_response = requests.post(
        f"{stage1_url}/part-result",
        json={
            "part_index": part_index,
            "result": result,
        },
        timeout=300,
    )

    post_response.raise_for_status()

    print(
        f"Stage1 response: "
        f"{post_response.text}",
        flush=True,
    )

    final_total_seconds = round(
        time.perf_counter() - worker_start,
        4,
    )

    print(
        "TOTAL_COMPARE_WORKER_SECONDS_WITH_POST="
        f"{final_total_seconds}",
        flush=True,
    )

    print(
        "=== Compare worker completed ===",
        flush=True,
    )


if __name__ == "__main__":
    main()