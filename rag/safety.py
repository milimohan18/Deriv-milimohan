"""Lightweight prompt-injection defences. Docs and questions are untrusted input:
the evaluator (or anyone) can replace them, so text inside them is data, never instructions."""
import re

# Common injection phrasing: attempts to override rules, change role, leak the prompt,
# or smuggle chat-template / role markers into the context.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(instructions?|rules|prompts?|guidelines|polic(y|ies))\b",
    r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}\b(system prompt|your (instructions|prompt|rules))\b",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bnew instructions?\s*:",
    r"\bdo not (follow|obey)\b",
    r"^\s*(system|assistant|developer)\s*:",
    r"<\|?\s*(im_start|im_end|system|endoftext)\s*\|?>",
    r"</?\s*(question|context|chunk|instructions?|system)\b[^>]*>",  # forged prompt tags
    r"\bset\s+[\"']?supported[\"']?\s+to\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]


def find_injection(text: str) -> list[str]:
    """Return the suspicious snippets found in text (empty list = looks clean)."""
    return [m.group(0).strip() for rx in _COMPILED for m in [rx.search(text)] if m]


def escape_for_prompt(text: str) -> str:
    """Neutralise angle brackets so untrusted text cannot open/close the prompt's <chunk> tags."""
    return text.replace("<", "&lt;").replace(">", "&gt;")
