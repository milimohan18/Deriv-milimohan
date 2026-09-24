"""Explicit pipeline stages and a tracker that records each transition."""
import json
from datetime import datetime, timezone
from enum import Enum

from . import config


class Stage(str, Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    CHUNKS_CREATED = "CHUNKS_CREATED"
    INDEX_BUILT = "INDEX_BUILT"
    QUESTIONS_LOADED = "QUESTIONS_LOADED"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    CITATIONS_VALIDATED = "CITATIONS_VALIDATED"
    RESULTS_EXPORTED = "RESULTS_EXPORTED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"


ORDER = list(Stage)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageTracker:
    """Enforces that stages happen in the required order and logs them to disk."""

    def __init__(self):
        self.history = []
        self.current = None
        self.advance(Stage.INIT, "pipeline started")

    def advance(self, stage: Stage, detail: str = "", artifact=None):
        expected = ORDER[0] if self.current is None else ORDER[ORDER.index(self.current) + 1]
        if stage != expected:
            raise RuntimeError(f"Stage order violated: expected {expected}, got {stage}")
        self.current = stage
        entry = {"stage": stage.value, "timestamp": now_iso(), "detail": detail}
        if artifact:
            entry["artifact"] = str(artifact.relative_to(config.ROOT).as_posix())
        self.history.append(entry)
        print(f"[{stage.value}] {detail}")
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        config.PIPELINE_STATE_PATH.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
