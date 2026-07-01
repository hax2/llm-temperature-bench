from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tempbench.config import PromptSpec

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def local_metrics(text: str, prompt: PromptSpec) -> dict[str, Any]:
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    bigrams = list(zip(words, words[1:]))
    repeated_bigram_ratio = _repetition_ratio(bigrams)
    trigrams = list(zip(words, words[1:], words[2:]))
    repeated_trigram_ratio = _repetition_ratio(trigrams)
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    checks = constraint_checks(text, prompt)
    return {
        "word_count": len(words),
        "character_count": len(text),
        "paragraph_count": len(paragraphs),
        "unique_word_ratio": round(unique_ratio, 6),
        "repeated_bigram_ratio": round(repeated_bigram_ratio, 6),
        "repeated_trigram_ratio": round(repeated_trigram_ratio, 6),
        "constraint_checks": checks,
        "all_machine_constraints_pass": all(checks.values()) if checks else None,
    }


def _repetition_ratio(ngrams: list[tuple[str, ...]]) -> float:
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def constraint_checks(text: str, prompt: PromptSpec) -> dict[str, bool]:
    constraints = prompt.constraints
    checks: dict[str, bool] = {}
    for heading in constraints.get("required_headings", []):
        checks[f"contains_heading:{heading}"] = heading in text
    lowered = text.lower()
    for phrase in constraints.get("forbidden_phrases", []):
        checks[f"omits_phrase:{phrase}"] = phrase.lower() not in lowered
    return checks
