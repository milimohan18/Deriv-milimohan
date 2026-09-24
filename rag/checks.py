"""Artifact-level checks shared by run.py (stage VALIDATION_COMPLETE) and validate.py."""
import json

from . import config

REQUIRED = [
    config.QUESTIONS_PATH,
    config.DOCUMENTS_PATH,
    config.CHUNKS_PATH,
    config.RETRIEVAL_PATH,
    config.ANSWERS_PATH,
    config.CITATION_VALIDATION_PATH,
    config.FINAL_ANSWERS_PATH,
]


def _rel(p):
    return p.relative_to(config.ROOT).as_posix()


def run_checks() -> list[tuple[str, bool, str]]:
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        return ok

    # 1. required files exist
    missing = [_rel(p) for p in REQUIRED if not p.is_file()]
    if not check("required artifact files exist", not missing, f"missing: {missing}" if missing else ""):
        return results

    # 2. JSON is valid
    data, bad = {}, []
    for p in REQUIRED:
        try:
            data[p] = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            bad.append(f"{_rel(p)}: {exc}")
    if not check("all artifacts are valid JSON", not bad, "; ".join(bad)):
        return results

    questions = data[config.QUESTIONS_PATH]
    documents = data[config.DOCUMENTS_PATH]
    chunks = data[config.CHUNKS_PATH]
    retrieval = data[config.RETRIEVAL_PATH]
    answers = data[config.ANSWERS_PATH]
    final = data[config.FINAL_ANSWERS_PATH]

    # LLM log (only required if an LLM was used)
    llm_used = any(str(a.get("generation_mode", "")).startswith("llm") for a in answers)
    if llm_used or config.LLM_CALLS_PATH.exists():
        bad_lines = []
        if config.LLM_CALLS_PATH.exists():
            for i, line in enumerate(config.LLM_CALLS_PATH.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    rec = json.loads(line)
                    need = {"stage", "timestamp", "provider", "model", "question_id", "prompt_hash",
                            "input_artifacts", "output_artifact"}
                    if need - rec.keys():
                        bad_lines.append(f"line {i} missing {sorted(need - rec.keys())}")
                except json.JSONDecodeError:
                    bad_lines.append(f"line {i} invalid JSON")
        else:
            bad_lines.append("llm_calls.jsonl missing although LLM answers exist")
        check("llm_calls.jsonl is valid", not bad_lines, "; ".join(bad_lines))

    # 3. chunk schema and offsets map back to the source documents
    doc_text = {d["doc_id"]: d["text"] for d in documents}
    chunk_ids = [c["chunk_id"] for c in chunks]
    bad_chunks = [
        c["chunk_id"] for c in chunks
        if c["doc_id"] not in doc_text or doc_text[c["doc_id"]][c["start_char"]:c["end_char"]] != c["text"]
    ]
    check("chunk IDs unique", len(chunk_ids) == len(set(chunk_ids)))
    check("chunk offsets match source documents", not bad_chunks, f"bad: {bad_chunks}")

    # 4. every question has exactly one answer (and one retrieval / final record)
    q_ids = [q["id"] for q in questions]
    for name, records in (("answers.json", answers), ("retrieval_results.json", retrieval),
                          ("final_answers.json", final)):
        ids = [r["id"] for r in records]
        dupes = sorted({i for i in ids if ids.count(i) > 1}, key=str)
        missing_q = [i for i in q_ids if i not in ids]
        extra = [i for i in ids if i not in q_ids]
        check(f"exactly one record per question in {name}", not (dupes or missing_q or extra),
              f"duplicates={dupes} missing={missing_q} extra={extra}")

    # 5. at most TOP_K retrieved chunks, all of which exist
    valid_chunks = set(chunk_ids)
    too_many = [r["id"] for r in retrieval if len(r["retrieved_chunks"]) > config.TOP_K]
    unknown = [r["id"] for r in retrieval if any(c["chunk_id"] not in valid_chunks for c in r["retrieved_chunks"])]
    check(f"each question has at most {config.TOP_K} retrieved chunks", not too_many, f"ids: {too_many}")
    check("retrieved chunk IDs exist in chunks.json", not unknown, f"ids: {unknown}")
    by_chunk = {c["chunk_id"]: c for c in chunks}
    altered = [
        f"{r['id']}:{c['chunk_id']}" for r in retrieval for c in r["retrieved_chunks"]
        if c["chunk_id"] in by_chunk
        and (c["text"] != by_chunk[c["chunk_id"]]["text"] or c["doc_id"] != by_chunk[c["chunk_id"]]["doc_id"])
    ]
    check("retrieved chunk text/doc_id identical to chunks.json", not altered, f"altered: {altered}")

    # 6. citations are valid and consistent with 'supported'
    retrieved_by_q = {r["id"]: {c["chunk_id"] for c in r["retrieved_chunks"]} for r in retrieval}
    bad_cites = [a["id"] for a in answers if any(c not in retrieved_by_q.get(a["id"], set()) for c in a["citations"])]
    unsup_cited = [a["id"] for a in answers if not a["supported"] and a["citations"]]
    sup_uncited = [a["id"] for a in answers if a["supported"] and not a["citations"]]
    check("cited chunk IDs are among the question's retrieved chunks", not bad_cites, f"ids: {bad_cites}")
    check("unsupported answers have no citations", not unsup_cited, f"ids: {unsup_cited}")
    check("supported answers have citations", not sup_uncited, f"ids: {sup_uncited}")

    return results

