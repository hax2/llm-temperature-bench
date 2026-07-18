from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "sampler-pilot" / "reports" / "all_results.csv"
DESTINATION = ROOT / "study-site" / "data" / "results.json"
NUMBER_FIELDS = {
    "sample": int,
    "seed": int,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "word_count": int,
    "generated_token_count": int,
    "unique_word_ratio": float,
    "repeated_bigram_ratio": float,
    "repeated_trigram_ratio": float,
    "overall_score": float,
    "coherence_score": float,
    "factuality_score": float,
    "creativity_score": float,
    "internal_consistency_score": float,
    "instruction_following_score": float,
    "fluency_score": float,
}


def convert(row: dict[str, str]) -> dict[str, object]:
    converted: dict[str, object] = dict(row)
    for field, caster in NUMBER_FIELDS.items():
        value = row.get(field, "")
        converted[field] = caster(value) if value else None
    for field in ("hit_token_cap", "judged", "machine_constraints_pass"):
        value = row.get(field, "")
        converted[field] = value.lower() == "true" if value else None
    return converted


with SOURCE.open(newline="", encoding="utf-8") as handle:
    rows = [convert(row) for row in csv.DictReader(handle)]

if len(rows) != 288:
    raise SystemExit(f"Expected 288 rows, found {len(rows)}")
if len({(row["model_id"], row["prompt_id"], row["temperature"], row["sampling_profile"]) for row in rows}) != 288:
    raise SystemExit("Sampler-pilot rows do not form 288 unique conditions")
if not all(row["judged"] for row in rows):
    raise SystemExit("Sampler-pilot contains unjudged rows")

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
DESTINATION.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {len(rows)} sampler-pilot rows to {DESTINATION}")
