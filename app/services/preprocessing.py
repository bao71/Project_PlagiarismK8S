import re
import unicodedata
from io import BytesIO
from dataclasses import dataclass

import pdfplumber
from underthesea import sent_tokenize


@dataclass
class SentenceRecord:
    page_number: int
    sentence_index: int
    sentence_index_page: int
    sentence_text: str

    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float


MIN_WORDS = 8

HEADER_HEIGHT = 70
FOOTER_HEIGHT = 70

NOISE_PATTERNS = [
    r"nhận xét của giảng viên",
    r"giảng viên hướng dẫn",
    r"mục lục",
    r"danh mục hình",
    r"danh mục bảng",
    r"lời cảm ơn",
    r"tài liệu tham khảo",
    r"^hình\s+\d+",
    r"^bảng\s+\d+",
    r"^figure\s+\d+",
    r"^table\s+\d+",
]


def clean_text(text: str) -> str:

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\.{3,}", " ", text)

    text = re.sub(
        r"[^a-zA-ZÀ-ỹ0-9\s\.\!\?]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"^\s*\d+\s*$",
        " ",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

def remove_heading_prefix(sentence: str) -> str:


    sentence = sentence.strip()

    sentence = re.sub(
        r"^chương\s+\d+\s*[\:\.]?",
        "",
        sentence,
        flags=re.IGNORECASE
    ).strip()

    sentence = re.sub(
        r"^\d+(\.\d+)+",
        "",
        sentence
    ).strip()

    words = sentence.split()

    if len(words) <= 6 and "." not in sentence:
        return ""

    return sentence.strip()

def _is_valid_sentence(
    sentence: str,
    min_words: int = MIN_WORDS
) -> bool:

    sentence = sentence.strip()

    if not sentence:
        return False

    if len(sentence.split()) < min_words:
        return False

    if re.fullmatch(r"[\W\d_\.]+", sentence):
        return False

    if re.search(r"\.{5,}", sentence):
        return False

    if re.search(r"\.{3,}\s*\d+", sentence):
        return False

    if re.match(r"^hình\s+\d+", sentence):
        return False

    if re.match(r"^bảng\s+\d+", sentence):
        return False

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, sentence):
            return False

    digit_count = sum(c.isdigit() for c in sentence)

    if digit_count / max(len(sentence), 1) > 0.2:
        return False

    letters = sum(c.isalpha() for c in sentence)

    if letters / max(len(sentence), 1) < 0.5:
        return False

    return True

def _compute_bbox_for_sentence(
    sent_words: list[str],
    page_words: list[dict],
    word_cursor: int,
    fallback_bbox: tuple[float, float, float, float],
):

    matched = []

    cursor = word_cursor

    for sw in sent_words:

        sw_clean = clean_text(sw)

        while cursor < len(page_words):

            pw_clean = clean_text(page_words[cursor]["text"])

            if (
                pw_clean == sw_clean
                or sw_clean in pw_clean
                or pw_clean in sw_clean
            ):
                matched.append(page_words[cursor])
                cursor += 1
                break

            cursor += 1

    if matched:

        x0 = min(w["x0"] for w in matched)
        y0 = min(w["top"] for w in matched)
        x1 = max(w["x1"] for w in matched)
        y1 = max(w["bottom"] for w in matched)

        return (
            round(x0, 2),
            round(y0, 2),
            round(x1, 2),
            round(y1, 2),
        ), cursor

    return fallback_bbox, cursor


def _process_page(
    page,
    page_number: int,
    global_index: int,
):

    page_height = page.height
    page_width = page.width

    cropped_page = page.crop((
        0,
        HEADER_HEIGHT,
        page_width,
        page_height - FOOTER_HEIGHT
    ))

    page_words = cropped_page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        extra_attrs=["size"],
    )

    if not page_words:
        return [], "", global_index


    raw_text = " ".join(w["text"] for w in page_words)

    cleaned_text = clean_text(raw_text)

    raw_sentences = sent_tokenize(cleaned_text)

    valid_sentences = []

    for s in raw_sentences:

        s = remove_heading_prefix(s)

        s = s.strip()

        if not s:
            continue

        if _is_valid_sentence(s):
            valid_sentences.append(s)

    fallback_bbox = (
        round(min(w["x0"] for w in page_words), 2),
        round(min(w["top"] for w in page_words), 2),
        round(max(w["x1"] for w in page_words), 2),
        round(max(w["bottom"] for w in page_words), 2),
    )

    records = []

    word_cursor = 0

    for sent_idx, sent in enumerate(valid_sentences):

        sent_words = sent.split()

        bbox, word_cursor = _compute_bbox_for_sentence(
            sent_words,
            page_words,
            word_cursor,
            fallback_bbox
        )

        records.append(
            SentenceRecord(
                page_number=page_number,
                sentence_index=global_index,
                sentence_index_page=sent_idx,
                sentence_text=sent,

                bbox_x0=bbox[0],
                bbox_y0=bbox[1],
                bbox_x1=bbox[2],
                bbox_y1=bbox[3],
            )
        )

        global_index += 1

    return records, cleaned_text, global_index


def extract_and_preprocess(
    pdf_bytes: BytesIO,
):

    all_sentences = []

    full_text_parts = []

    global_index = 0

    with pdfplumber.open(pdf_bytes) as pdf:

        for page_num, page in enumerate(pdf.pages, start=1):

            records, cleaned_page_text, global_index = _process_page(
                page,
                page_num,
                global_index
            )

            if cleaned_page_text:
                full_text_parts.append(cleaned_page_text)

            all_sentences.extend(records)

    full_text_cleaned = " ".join(full_text_parts)

    return full_text_cleaned, all_sentences