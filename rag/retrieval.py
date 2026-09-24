"""Stages INDEX_BUILT and RETRIEVAL_COMPLETE: BM25 over chunks, built at runtime."""
import math
from collections import Counter

from . import config
from .chunking import is_heading, paragraph_spans
from .io_utils import write_json
from .text_utils import tokenize


def heading_context(doc_text: str, start_char: int) -> str:
    """Parent headings of a chunk, e.g. 'Authentication > API keys'.

    Walks the headings in order keeping a stack by level ('#' count; in .txt files the
    first title line is level 1, later ones level 2). A heading replaces earlier headings of
    the same or deeper level - including one that opens the chunk itself, so a chunk starting
    with '## OAuth 2.0' inherits only its document title, not the previous section.
    """
    stack = []  # (level, heading text)
    for s, e in paragraph_spans(doc_text):
        if s > start_char:
            break
        if not is_heading(doc_text, (s, e)):
            continue
        block = doc_text[s:e].strip()
        hashes = len(block) - len(block.lstrip("#"))
        level = hashes or (1 if not stack else 2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if s < start_char:  # the chunk's own opening heading is already in its text
            stack.append((level, block.lstrip("#").strip()))
    return " > ".join(text for _, text in stack)


def search_text(chunk: dict, doc_texts: dict) -> str:
    """Text used for BM25 scoring only; the chunk's saved text/offsets are never modified."""
    context = heading_context(doc_texts.get(chunk["doc_id"], ""), chunk["start_char"])
    return f"{context}\n{chunk['text']}" if context else chunk["text"]


class BM25Index:
    def __init__(self, chunks: list[dict], documents: list[dict] | None = None,
                 k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.chunks = chunks
        self.k1, self.b = k1, b
        doc_texts = {d["doc_id"]: d["text"] for d in documents or []}
        self.search_texts = [search_text(c, doc_texts) for c in chunks]
        self.term_freqs = [Counter(tokenize(t)) for t in self.search_texts]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(chunks)
        # BM25 IDF with +1 so it is always positive
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}
        # Weight for question terms absent from the corpus (used by the rule-based answerer):
        # treat them like the rarest term that does occur, so one missing word can't dominate.
        self.max_idf = max(self.idf.values(), default=0.0)
        # terms of each chunk's own text (no headings), for the meaningful-match check
        self.text_terms = [set(tokenize(c["text"])) for c in chunks]

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, self.max_idf)

    def score(self, query_terms: list[str], i: int) -> float:
        tf, length = self.term_freqs[i], self.lengths[i]
        total = 0.0
        for term in set(query_terms):
            f = tf.get(term, 0)
            if f:
                norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[term] * f * (self.k1 + 1) / (f + norm)
        return total

    def coverage(self, query_terms: set[str], i: int) -> float:
        """IDF-weighted share of the question's terms found in the chunk's own text."""
        total = sum(self.term_idf(t) for t in query_terms)
        found = sum(self.term_idf(t) for t in query_terms & self.text_terms[i])
        return found / total if total else 0.0

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        terms = tokenize(query)
        scored = [(self.score(terms, i), i) for i in range(len(self.chunks))]
        # 1. top-k candidates: score desc, chunk order for deterministic ties, zero scores dropped
        scored = sorted((s for s in scored if s[0] > 0), key=lambda x: (-x[0], x[1]))[:top_k]
        # 2. relative cutoff: no weak filler just to reach top_k
        if scored:
            top_score = scored[0][0]
            scored = [(s, i) for s, i in scored if s >= config.RELATIVE_SCORE_CUTOFF * top_score]
        # 3. meaningful match: the chunk text itself must cover enough of the question
        scored = [(s, i) for s, i in scored if self.coverage(set(terms), i) >= config.MIN_QUERY_COVERAGE]
        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "doc_id": self.chunks[i]["doc_id"],
                "score": round(s, 4),
                "text": self.chunks[i]["text"],
            }
            for s, i in scored
        ]


def build_index(chunks: list[dict], documents: list[dict]) -> BM25Index:
    return BM25Index(chunks, documents)


def retrieve_all(index: BM25Index, questions: list[dict]) -> list[dict]:
    results = [
        {"id": q["id"], "question": q["question"], "retrieved_chunks": index.search(q["question"])}
        for q in questions
    ]
    write_json(config.RETRIEVAL_PATH, results)
    return results
