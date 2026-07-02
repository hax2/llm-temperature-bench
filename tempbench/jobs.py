from __future__ import annotations

import hashlib
from collections.abc import Iterable

from tempbench.config import BenchmarkConfig, ModelSpec, PromptSpec
from tempbench.storage import Job


def select_models(config: BenchmarkConfig, requested: Iterable[str] | None) -> list[ModelSpec]:
    if not requested:
        return [model for model in config.models if model.enabled]
    return _select(config.models, requested, "model")


def select_prompts(config: BenchmarkConfig, requested: Iterable[str] | None) -> list[PromptSpec]:
    return _select(config.prompts, requested, "prompt")


def _select(items, requested, kind):
    if not requested:
        return list(items)
    wanted = set(requested)
    found = [item for item in items if item.id in wanted]
    missing = sorted(wanted - {item.id for item in found})
    if missing:
        raise ValueError(f"Unknown {kind} IDs: {', '.join(missing)}")
    return found


def create_jobs(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
) -> list[Job]:
    jobs = []
    for model in models:
        for temperature in config.run.temperatures:
            for prompt in prompts:
                for sample in range(1, config.run.samples_per_condition + 1):
                    # Excluding temperature couples the RNG stream across temperature
                    # conditions, reducing an avoidable source of sampling variance.
                    key = f"{model.id}|{prompt.id}|{sample}|{config.run.seed}"
                    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append(
                        Job(
                            model_id=model.id,
                            prompt_id=prompt.id,
                            temperature=temperature,
                            sample=sample,
                            seed=(config.run.seed + offset) % (2**31),
                        )
                    )
    return jobs
