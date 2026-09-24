"""Stage CITATIONS_VALIDATED: deterministic checks on every generated answer."""
from . import config
from .io_utils import write_json
from .safety import find_injection
from .text_utils import extract_numbers, tokenize


def grounding_issues(answer: str, cited_texts: list[str]) -> tuple[list[str], dict]:
    """Two lightweight grounding checks against the cited chunk text."""
    issues = []
    cited = "\n".join(cited_texts)

    # A. every numeric claim in the answer must appear in the cited text
    missing_numbers = sorted(extract_numbers(answer) - extract_numbers(cited))
    if missing_numbers:
        issues.append(f"numeric claims not found in cited chunks: {missing_numbers}")

    # B. most answer content words must appear in the cited text
    answer_terms = set(tokenize(answer))
    cited_terms = set(tokenize(cited))
    overlap = len(answer_terms & cited_terms) / len(answer_terms) if answer_terms else 1.0
    if overlap < config.GROUNDING_MIN_OVERLAP:
        issues.append(
            f"low term overlap with cited text ({overlap:.2f} < {config.GROUNDING_MIN_OVERLAP})"
        )
    return issues, {"term_overlap": round(overlap, 3), "unsupported_numbers": missing_numbers}


def validate_answer(answer: dict, retrieval: dict) -> dict:
    issues = []
    retrieved = {c["chunk_id"]: c["text"] for c in retrieval["retrieved_chunks"]}
    citations = answer.get("citations", [])

    invalid = [c for c in citations if c not in retrieved]
    for cid in invalid:
        issues.append(f"cited chunk '{cid}' was not retrieved for this question")

    grounding = None
    if answer["supported"]:
        if not citations:
            issues.append("supported answer has no citations")
        valid_texts = [retrieved[c] for c in citations if c in retrieved]
        if valid_texts:
            g_issues, grounding = grounding_issues(answer["answer"], valid_texts)
            issues.extend(g_issues)
    elif citations:
        issues.append("unsupported answer must have empty citations")

    # Prompt-injection checks: evidence must not come from suspicious text, and the answer
    # must not echo injected instructions or leak the prompt.
    for cid in citations:
        if cid in retrieved and find_injection(retrieved[cid]):
            issues.append(f"cited chunk '{cid}' contains suspected prompt-injection text")
    if find_injection(answer["answer"]):
        issues.append(f"answer contains suspected injection/prompt-leak text: {find_injection(answer['answer'])}")
    warnings = []
    if find_injection(retrieval.get("question", "")):
        warnings.append(f"question contains suspected prompt-injection text: {find_injection(retrieval['question'])}")

    result = {"id": answer["id"], "passed": not issues, "issues": issues}
    if warnings:
        result["warnings"] = warnings
    if grounding is not None:
        result["grounding"] = grounding
    return result


def validate_citations(answers: list[dict], retrieval_results: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in retrieval_results}
    results = []
    for a in answers:
        retrieval = by_id.get(a["id"])
        if retrieval is None:
            results.append({"id": a["id"], "passed": False, "issues": ["no retrieval result for question"]})
        else:
            results.append(validate_answer(a, retrieval))
    write_json(config.CITATION_VALIDATION_PATH, results)
    return results
