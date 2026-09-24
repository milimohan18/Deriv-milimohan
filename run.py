"""Run the full RAG pipeline end to end: python run.py"""
import sys

from rag import config
from rag.checks import run_checks
from rag.chunking import create_chunks
from rag.generation import generate_answers
from rag.ingest import load_documents, load_questions
from rag.io_utils import write_json
from rag.retrieval import build_index, retrieve_all
from rag.stages import Stage, StageTracker
from rag.validation import validate_citations


def clean_artifacts() -> None:
    """Remove previous outputs so every artifact is regenerated from the inputs."""
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    for p in config.ARTIFACTS_DIR.iterdir():
        if p.is_file() and p.suffix in {".json", ".jsonl"}:
            p.unlink()


def export_final(questions, retrieval, answers, validations) -> list[dict]:
    retr = {r["id"]: r for r in retrieval}
    ans = {a["id"]: a for a in answers}
    val = {v["id"]: v for v in validations}
    final = []
    for q in questions:
        a, v = ans[q["id"]], val[q["id"]]
        final.append({
            "id": q["id"],
            "question": q["question"],
            "answer": a["answer"],
            "supported": a["supported"],
            "citations": a["citations"],
            "retrieved_chunk_ids": [c["chunk_id"] for c in retr[q["id"]]["retrieved_chunks"]],
            "generation_mode": a["generation_mode"],
            "validation": {"passed": v["passed"], "issues": v["issues"]},
        })
    write_json(config.FINAL_ANSWERS_PATH, final)
    return final


def main() -> int:
    clean_artifacts()
    tracker = StageTracker()

    documents = load_documents()
    tracker.advance(Stage.DOCUMENTS_LOADED, f"{len(documents)} documents", config.DOCUMENTS_PATH)

    chunks = create_chunks(documents)
    tracker.advance(Stage.CHUNKS_CREATED, f"{len(chunks)} chunks (CHUNK_SIZE={config.CHUNK_SIZE})", config.CHUNKS_PATH)

    index = build_index(chunks, documents)
    tracker.advance(Stage.INDEX_BUILT, f"BM25 index over {len(chunks)} chunks, {len(index.idf)} terms")

    questions = load_questions()
    tracker.advance(Stage.QUESTIONS_LOADED, f"{len(questions)} questions", config.QUESTIONS_PATH)

    retrieval = retrieve_all(index, questions)
    n_retrieved = sum(len(r["retrieved_chunks"]) for r in retrieval)
    empty = sum(not r["retrieved_chunks"] for r in retrieval)
    tracker.advance(Stage.RETRIEVAL_COMPLETE,
                    f"{n_retrieved} chunks retrieved (up to {config.TOP_K} per question, {empty} with no meaningful match)",
                    config.RETRIEVAL_PATH)

    answers, mode = generate_answers(retrieval, index.term_idf)
    tracker.advance(Stage.ANSWERS_GENERATED, f"{len(answers)} answers (mode={mode})", config.ANSWERS_PATH)

    validations = validate_citations(answers, retrieval)
    failures = sum(not v["passed"] for v in validations)
    tracker.advance(Stage.CITATIONS_VALIDATED, f"{failures} answers failed validation", config.CITATION_VALIDATION_PATH)

    export_final(questions, retrieval, answers, validations)
    tracker.advance(Stage.RESULTS_EXPORTED, "final answers written", config.FINAL_ANSWERS_PATH)

    checks = run_checks()
    failed_checks = [name for name, ok, _ in checks if not ok]
    tracker.advance(Stage.VALIDATION_COMPLETE, f"{len(checks) - len(failed_checks)}/{len(checks)} artifact checks passed")

    print("\n===== Summary =====")
    print(f"Documents:            {len(documents)}")
    print(f"Chunks:               {len(chunks)}")
    print(f"Questions:            {len(questions)}")
    print(f"Supported answers:    {sum(a['supported'] for a in answers)}")
    print(f"Validation failures:  {failures}")
    print(f"Answer mode:          {mode}")
    if failed_checks:
        print(f"Artifact checks FAILED: {failed_checks}")
    return 1 if failed_checks else 0


if __name__ == "__main__":
    sys.exit(main())
