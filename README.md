# Grounded KB Assistant (small RAG pipeline)

Answers questions from a local knowledge base (`docs/`) with citations to retrieved chunks,
refuses when the docs don't support an answer, and validates every answer deterministically.

**Pure Python 3.10+ standard library — no `pip install` required.**

## Run

```bash
python run.py        # full pipeline: regenerates everything in artifacts/
python validate.py   # re-checks the artifacts on disk (exit code 1 on failure)
```

Optional LLM (Google Gemini):

```bash
export GEMINI_API_KEY=...            # or put GEMINI_API_KEY=... in a .env file (git-ignored)
export GEMINI_MODEL=gemini-3.6-flash # optional, this is the default
export ANSWER_MODE=auto              # auto (default) | llm | rules
```

Without a key the pipeline still runs end to end using the rule-based answerer (below).

## Inputs

- `docs/` — any `.md` / `.txt` files (searched recursively; filenames don't matter)
- `questions.json` — `[{"id": ..., "question": "..."}]`

Both can be swapped for equivalent fixtures; nothing is hardcoded to the sample content.

## Pipeline stages

`run.py` executes each stage explicitly; `rag/stages.py` enforces the order and logs every
transition to `artifacts/pipeline_state.json`.

| Stage | Code | Output |
|---|---|---|
| INIT | `run.py` (clears old artifacts) | `pipeline_state.json` |
| DOCUMENTS_LOADED | `rag/ingest.py` | `documents.json` |
| CHUNKS_CREATED | `rag/chunking.py` | `chunks.json` |
| INDEX_BUILT | `rag/retrieval.py` (`BM25Index`) | in memory |
| QUESTIONS_LOADED | `rag/ingest.py` | — |
| RETRIEVAL_COMPLETE | `rag/retrieval.py` | `retrieval_results.json` |
| ANSWERS_GENERATED | `rag/generation.py` | `answers.json`, `llm_calls.jsonl` |
| CITATIONS_VALIDATED | `rag/validation.py` | `citation_validation.json` |
| RESULTS_EXPORTED | `run.py` | `final_answers.json` |
| VALIDATION_COMPLETE | `rag/checks.py` | printed summary |

All settings (chunk size, top-k, BM25 params, thresholds, model) live in `rag/config.py`.

## How each step works

**Ingestion.** Every `.md`/`.txt` under `docs/` is read (sorted, CRLF normalised).
`doc_id` is derived from the relative path (`guides/sdk.txt` → `guides_sdk`).

**Chunking.** `CHUNK_SIZE = 400` characters. Paragraphs (blank-line separated) are packed
greedily into chunks up to that size; paragraphs longer than the limit are split with a
sliding window (`CHUNK_OVERLAP = 80`) that prefers sentence boundaries. Headings are never
left dangling at the end of a chunk. IDs are `"{doc_id}__chunk_{n}"` (stable for the same
input), and `text == document_text[start_char:end_char]` always holds.

**Retrieval.** Hand-written BM25 (`k1=1.5`, `b=0.75`) over tokenised chunks (lowercase,
stopwords removed, light stemming). Scores are raw BM25 values.

- *Heading context (index only):* each chunk is scored on `parent headings + chunk text`,
  e.g. the "API keys" section is indexed as `Authentication` + its text, so it still matches
  "authentication" after chunking separated it from the document title. Parent headings
  follow the heading hierarchy (a chunk opening with `## OAuth 2.0` inherits only
  `# Authentication`). The saved chunk `text`, offsets and IDs are never modified.
- *Up to 3, no filler:* take the top `TOP_K = 3` candidates, then drop any scoring below
  `RELATIVE_SCORE_CUTOFF = 0.3` × the top score, and any whose **own text** (headings
  excluded) contains less than `MIN_QUERY_COVERAGE = 0.5` of the question's IDF-weighted
  terms. Headings can help a chunk rank but cannot make it relevant on their own. If nothing
  qualifies, `retrieved_chunks` is `[]` and the answer is unsupported without an LLM call.

**Answer generation.** Uses only the retrieved chunks for that question.

- *LLM mode* — Gemini, temperature 0, JSON output enforced via response schema. The system
  prompt (`SYSTEM_PROMPT` in `rag/generation.py`) instructs the model to answer only from
  the context, return the exact "not enough information" answer with `supported=false` and
  no citations when unsupported, cite the chunk IDs it used, and never invent features or
  policies. Each call is logged to `artifacts/llm_calls.jsonl` (stage, timestamp, provider,
  model, question_id, SHA-256 prompt hash, input/output artifacts, status, latency, raw
  response). HTTP 429/5xx are retried with backoff; if a call still fails, that question
  falls back to rule-based mode (`generation_mode: "rules_fallback"`).
- *Rule-based mode* (no key) — deterministic extractive answering:
  1. Tokenise the question; weight each term by its BM25 IDF (terms absent from the corpus
     get the weight of the rarest corpus term).
  2. Split retrieved chunks into sentences; greedily pick up to 2 sentences that cover the
     most not-yet-covered question weight.
  3. If the picked sentences cover ≥ 60% of total question weight (`RULE_MIN_COVERAGE`),
     the answer is those sentences verbatim, cited with their chunk IDs. Otherwise the answer
     is `"The knowledge base does not provide enough information to answer this."`,
     `supported=false`, `citations=[]`.

Each answer records its `generation_mode` (`llm:<model>`, `rules`, `rules_fallback`,
`no_context`).

**Citation validation** (deterministic, after generation), per answer:
- every cited chunk ID must be in that question's retrieved chunks;
- `supported=true` ⇒ at least one citation; `supported=false` ⇒ no citations;
- grounding A: every number in the answer (`30`, `$49`, `1,000,000`) must appear in the cited text;
- grounding B: ≥ 50% of the answer's content words must appear in the cited text.

Failures are recorded as `issues`; they do not stop the pipeline.

**Artifact checks** (`validate.py`, also run as the last stage): required files exist, JSON
is valid, `llm_calls.jsonl` records are complete, chunk offsets match the source documents,
exactly one answer/retrieval/final record per question, ≤ 3 retrieved chunks per question,
cited IDs are valid, supported answers have citations, unsupported answers have none.

## Output example (`artifacts/final_answers.json`)

```json
{
  "id": 6,
  "question": "Does the platform support GraphQL subscriptions?",
  "answer": "The knowledge base does not provide enough information to answer this.",
  "supported": false,
  "citations": [],
  "retrieved_chunk_ids": ["authentication__chunk_2", "authentication__chunk_0", "pricing__chunk_1"],
  "generation_mode": "llm:gemini-3.6-flash",
  "validation": {"passed": true, "issues": []}
}
```

## Known limitations

- BM25 and the rule-based answerer are lexical: synonyms ("languages" vs "Python, Go") are
  not matched, so rules mode may refuse answerable questions. It fails safe (refuses rather
  than guesses); LLM mode handles these better.
- Grounding checks are heuristics: they catch invented numbers and off-topic text, not
  subtle paraphrase errors. In particular, a fact moved between subjects *and* cited to the
  chunk it came from (e.g. "webhook logs are kept 90 days" citing the chunk about API request
  logs) passes the deterministic checks; the prompt rule against subject transfer and the
  retrieval filters are the defence there.
- Retrieval can still return a lexically similar chunk about a neighbouring subject (Q2
  returns the general retention-periods chunk alongside the webhook-logs chunk).
- LLM output can vary slightly between runs even at temperature 0; free-tier rate limits may
  trigger the rule-based fallback for some questions.
