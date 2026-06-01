import asyncio
import json
import os
from typing import Any, Dict

import requests
import time
from app.services import embedding
from app.services.checker import run_plagiarism_check
from app.services.preprocessing import SentenceRecord


def convert_check_response_to_dict(check_response: Any) -> Dict[str, Any]:
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
) -> list[SentenceRecord]:
    sentence_records = []

    for index, sentence in enumerate(sentences):
        sentence_text = sentence.get("sentence_text")

        if not sentence_text:
            raise ValueError(
                f"Missing sentence_text at index={index}: {sentence}"
            )

        sentence_index = int(
            sentence.get("sentence_index", index)
        )

        normalized = {
            "page_number": int(sentence.get("page_number", 0)),
            "sentence_index": sentence_index,
            "sentence_index_page": int(
                sentence.get("sentence_index_page", sentence_index)
            ),
            "sentence_text": sentence_text,
            "bbox_x0": float(sentence.get("bbox_x0", 0.0)),
            "bbox_y0": float(sentence.get("bbox_y0", 0.0)),
            "bbox_x1": float(sentence.get("bbox_x1", 0.0)),
            "bbox_y1": float(sentence.get("bbox_y1", 0.0)),
        }

        sentence_records.append(
            SentenceRecord(**normalized)
        )

    return sentence_records

def main():
    worker_start = time.perf_counter()

    stage1_url = os.getenv("STAGE1_URL")
    part_index = int(os.getenv("PART_INDEX", "0"))
    part_count = int(os.getenv("PART_COUNT", "1"))

    if not stage1_url:
        raise ValueError("Missing STAGE1_URL")

    print("=== Start compare worker ===", flush=True)
    print(f"STAGE1_URL={stage1_url}", flush=True)
    print(f"PART_INDEX={part_index}", flush=True)
    print(f"PART_COUNT={part_count}", flush=True)

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

    raw_sentences = payload["sentences"]
    candidates = payload["candidates"]

    print(
        f"Loaded sentences={len(raw_sentences)}, "
        f"candidates={len(candidates)}",
        flush=True,
    )

    sentence_records = convert_sentences_to_records(
        raw_sentences
    )

    if not candidates:
        result = {
            "total_sentences": len(sentence_records),
            "plagiarized_sentences": 0,
            "plagiarism_ratio": 0.0,
            "is_plagiarized": False,
            "sentence_labels": [0] * len(sentence_records),
            "references": [],
        }

    else:
        sentence_texts = [
            s.sentence_text
            for s in sentence_records
        ]

        print("Generating embeddings...", flush=True)

        query_embeddings = embedding.embed_sentences(
            sentence_texts
        )

        print("Running plagiarism check...", flush=True)

        check_response = run_plagiarism_check(
            query_sentences=sentence_records,
            query_embeddings=query_embeddings,
            candidates=candidates,
        )

        result = convert_check_response_to_dict(
            check_response
        )

    total_compare_worker_seconds = round(
        time.perf_counter() - worker_start,
        4,
    )

    result["part_index"] = part_index
    result["part_count"] = part_count
    result["candidate_count"] = len(candidates)
    result["total_compare_worker_seconds"] = total_compare_worker_seconds

    print(
        f"TOTAL_COMPARE_WORKER_SECONDS={total_compare_worker_seconds}",
        flush=True,
    )

    print("Posting part result back to stage1...", flush=True)

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
        f"Stage1 response: {post_response.text}",
        flush=True,
    )

    final_total_seconds = round(
        time.perf_counter() - worker_start,
        4,
    )

    print(
        f"TOTAL_COMPARE_WORKER_SECONDS_WITH_POST={final_total_seconds}",
        flush=True,
    )

    print("=== Compare worker completed ===", flush=True)

if __name__ == "__main__":
    main()