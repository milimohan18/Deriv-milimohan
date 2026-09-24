"""Stage DOCUMENTS_LOADED and QUESTIONS_LOADED: read inputs from disk."""
import re
from pathlib import Path

from . import config
from .io_utils import read_json, write_json


def make_doc_id(rel_path: Path) -> str:
    """Stable id from the relative path, e.g. 'guides/webhooks.md' -> 'guides_webhooks'."""
    stem = rel_path.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list[dict]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    files = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in config.DOC_EXTENSIONS
    )
    documents = []
    for path in files:
        rel = path.relative_to(docs_dir)
        text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").strip()
        if not text:
            continue
        documents.append({
            "doc_id": make_doc_id(rel),
            "path": (Path(docs_dir.name) / rel).as_posix(),
            "text": text,
        })
    write_json(config.DOCUMENTS_PATH, documents)
    return documents


def load_questions(path: Path = config.QUESTIONS_PATH) -> list[dict]:
    questions = read_json(path)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON list")
    for q in questions:
        if "id" not in q or not str(q.get("question", "")).strip():
            raise ValueError(f"Each question needs 'id' and non-empty 'question': {q}")
    return questions

