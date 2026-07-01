from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tempbench.storage import RunStore

SCORE_NAMES = [
    "overall",
    "coherence",
    "factuality",
    "creativity",
    "internal_consistency",
    "instruction_following",
    "fluency",
]


def build_reports(store: RunStore) -> dict[str, Path]:
    store.reports.mkdir(parents=True, exist_ok=True)
    generations = {
        item["job"]["model_id"] + "|" + _job_suffix(item): item for item in store.iter_generations()
    }
    judgments = {
        item["job"]["model_id"] + "|" + _job_suffix(item): item for item in store.iter_judgments()
    }
    rows = [_flatten(generation, judgments.get(key)) for key, generation in generations.items()]
    rows.sort(
        key=lambda row: (row["model_id"], row["temperature"], row["prompt_id"], row["sample"])
    )

    raw_csv = store.reports / "all_results.csv"
    _write_csv(raw_csv, rows)
    summary_rows = _summarize(rows)
    summary_csv = store.reports / "summary_by_model_temperature.csv"
    _write_csv(summary_csv, summary_rows)
    html_path = store.reports / "report.html"
    html_path.write_text(_render_html(summary_rows, len(rows), len(judgments)), encoding="utf-8")
    return {"results_csv": raw_csv, "summary_csv": summary_csv, "html": html_path}


def _job_suffix(record: dict[str, Any]) -> str:
    job = record["job"]
    return f"{job['prompt_id']}|{job['temperature']}|{job['sample']}"


def _flatten(generation: dict[str, Any], judgment: dict[str, Any] | None) -> dict[str, Any]:
    job = generation["job"]
    local = generation.get("local_metrics", {})
    row: dict[str, Any] = {
        **job,
        "repo": generation["model"]["repo"],
        "variant": generation["model"]["variant"],
        "category": generation["prompt"]["category"],
        "word_count": local.get("word_count"),
        "generated_token_count": generation.get("runtime", {}).get("generated_token_count"),
        "hit_token_cap": generation.get("runtime", {}).get("hit_token_cap"),
        "unique_word_ratio": local.get("unique_word_ratio"),
        "repeated_bigram_ratio": local.get("repeated_bigram_ratio"),
        "repeated_trigram_ratio": local.get("repeated_trigram_ratio"),
        "machine_constraints_pass": local.get("all_machine_constraints_pass"),
        "output": generation["output"],
        "judged": judgment is not None,
    }
    scores = judgment.get("scores", {}) if judgment else {}
    for name in SCORE_NAMES:
        value = scores.get(name)
        row[f"{name}_score"] = value.get("score") if isinstance(value, dict) else None
    row["major_problems"] = json.dumps(scores.get("major_problems", []), ensure_ascii=False)
    return row


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["temperature"])].append(row)
    result = []
    for (model_id, temperature), members in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model_id": model_id,
            "temperature": temperature,
            "generations": len(members),
            "judged": sum(bool(row["judged"]) for row in members),
        }
        for name in SCORE_NAMES:
            values = [
                float(row[f"{name}_score"])
                for row in members
                if row.get(f"{name}_score") is not None
            ]
            summary[f"mean_{name}"] = round(statistics.fmean(values), 3) if values else None
        for metric in ["unique_word_ratio", "repeated_bigram_ratio", "repeated_trigram_ratio"]:
            values = [float(row[metric]) for row in members if row.get(metric) is not None]
            summary[f"mean_{metric}"] = round(statistics.fmean(values), 6) if values else None
        result.append(summary)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_html(rows: list[dict[str, Any]], generated: int, judged: int) -> str:
    columns = [
        "model_id",
        "temperature",
        "generations",
        "judged",
        "mean_overall",
        "mean_coherence",
        "mean_factuality",
        "mean_creativity",
        "mean_internal_consistency",
        "mean_instruction_following",
        "mean_fluency",
        "mean_repeated_trigram_ratio",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Temperature benchmark</title>
<style>
body{{font:14px system-ui,sans-serif;margin:2rem;color:#1f2937}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d1d5db;padding:.45rem}}
th{{background:#f3f4f6;position:sticky;top:0}}tr:nth-child(even){{background:#f9fafb}}
</style></head><body><h1>Temperature benchmark</h1>
<p>{generated} generations; {judged} Gemini judgments.</p>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body></html>"""
