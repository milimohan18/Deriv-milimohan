"""All tunable settings live here so behaviour is easy to inspect and change."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support (KEY=VALUE lines); real environment variables take precedence."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Inputs (may be replaced by the evaluator with equivalent fixtures)
DOCS_DIR = ROOT / "docs"
QUESTIONS_PATH = ROOT / "questions.json"
DOC_EXTENSIONS = {".md", ".txt"}

# Outputs
ARTIFACTS_DIR = ROOT / "artifacts"
DOCUMENTS_PATH = ARTIFACTS_DIR / "documents.json"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"
RETRIEVAL_PATH = ARTIFACTS_DIR / "retrieval_results.json"
ANSWERS_PATH = ARTIFACTS_DIR / "answers.json"
CITATION_VALIDATION_PATH = ARTIFACTS_DIR / "citation_validation.json"
FINAL_ANSWERS_PATH = ARTIFACTS_DIR / "final_answers.json"
LLM_CALLS_PATH = ARTIFACTS_DIR / "llm_calls.jsonl"
PIPELINE_STATE_PATH = ARTIFACTS_DIR / "pipeline_state.json"

# Chunking: target maximum characters per chunk, and overlap used only when a
# single paragraph is longer than CHUNK_SIZE and must be split mid-paragraph.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Retrieval
TOP_K = 3  # at most this many chunks per question (fewer if candidates are filtered out)
# Drop candidates scoring below this fraction of the best candidate's score (no filler).
RELATIVE_SCORE_CUTOFF = 0.3
# A chunk is a meaningful match only if its OWN text (headings excluded) contains at least
# this IDF-weighted share of the question's terms. If none qualify, retrieval returns [].
MIN_QUERY_COVERAGE = 0.5
BM25_K1 = 1.5
BM25_B = 0.75

# Answer generation
UNSUPPORTED_ANSWER = "The knowledge base does not provide enough information to answer this."
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 4  # retries on HTTP 429/5xx with exponential backoff
# Rule-based fallback: minimum IDF-weighted share of question terms that must
# appear in the best evidence sentences for the answer to count as supported.
RULE_MIN_COVERAGE = 0.6

# Grounding check: minimum share of answer content words found in cited text.
GROUNDING_MIN_OVERLAP = 0.5
