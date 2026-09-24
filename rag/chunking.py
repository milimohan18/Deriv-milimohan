"""Stage CHUNKS_CREATED: split documents into retrievable chunks with exact offsets."""
import re

from . import config
from .io_utils import write_json


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of blocks separated by blank lines."""
    spans = []
    for m in re.finditer(r"(?:[^\n]*\S[^\n]*(?:\n|$))+", text):
        start = m.start()
        end = start + len(m.group().rstrip())
        spans.append((start, end))
    return spans


def _split_long_span(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding window over one oversized paragraph, preferring sentence/word boundaries."""
    pieces = []
    s = start
    while s < end:
        e = min(s + size, end)
        if e < end:
            window = text[s:e]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut >= size // 2:
                e = s + cut + 1
            else:
                space = window.rfind(" ")
                if space > 0:
                    e = s + space
        pieces.append((s, e))
        if e >= end:
            break
        next_s = max(e - overlap, s + 1)
        # move forward to the start of a word
        while next_s < e and not text[next_s - 1].isspace():
            next_s += 1
        s = next_s
        while s < end and text[s].isspace():
            s += 1
    return pieces


def is_heading(text: str, span: tuple[int, int]) -> bool:
    """Markdown '#' headings, or short single-line blocks without end punctuation (.txt titles)."""
    block = text[span[0]:span[1]].strip()
    if block.startswith("#"):
        return True
    return "\n" not in block and len(block) <= 60 and not block.endswith((".", "!", "?", ":"))


def chunk_document(doc: dict, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    text = doc["text"]
    units = []
    for s, e in paragraph_spans(text):
        units.extend(_split_long_span(text, s, e, size, overlap) if e - s > size else [(s, e)])

    groups, current = [], []
    for unit in units:
        if current and unit[1] - current[0][0] > size:
            carry = []
            # don't leave headings dangling at the end of a chunk
            while len(current) > 1 and is_heading(text, current[-1]):
                carry.insert(0, current.pop())
            groups.append(current)
            current = carry
        current.append(unit)
    if current:
        groups.append(current)

    chunks = []
    for i, group in enumerate(groups):
        start, end = group[0][0], group[-1][1]
        chunks.append({
            "chunk_id": f"{doc['doc_id']}__chunk_{i}",
            "doc_id": doc["doc_id"],
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
        })
    return chunks


def create_chunks(documents: list[dict]) -> list[dict]:
    chunks = [c for doc in documents for c in chunk_document(doc)]
    write_json(config.CHUNKS_PATH, chunks)
    return chunks
