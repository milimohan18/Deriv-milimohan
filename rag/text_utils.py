"""Shared tokenisation used by retrieval, the rule-based answerer and grounding checks."""
import re

STOPWORDS = set("""
a an and are as at be been but by can could do does did for from has have how i if in
into is it its of on or our so such that the their then there these this to was we
were what when where which who why will with would you your my me about after before
any all also than them they those via per up out over under just only not no yes
long many much happen happens happened get
""".split())


def stem(token: str) -> str:
    """Very light suffix stripping so 'logs'/'log' and 'retained'/'retain' match."""
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith("ss"):
                return token
            if suffix == "ed" and token.endswith("eed"):  # exceed/need/speed are not past tense
                break
            token = token[: len(token) - len(suffix)] + repl
            break
    # drop a trailing 'e' so 'delete'/'deleted' -> 'delet'
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS]


def extract_numbers(text: str) -> set[str]:
    """Numeric claims normalised: '1,000,000' -> '1000000', '$49' -> '49', '99.95%' -> '99.95'."""
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip() and not p.strip().startswith("#")]
