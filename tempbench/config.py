from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(StrictModel):
    output_dir: Path = Path("results")
    temperatures: list[float]
    samples_per_condition: int = Field(default=1, ge=1)
    max_new_tokens: int = Field(default=900, ge=1)
    batch_size: int = Field(default=1, ge=1)
    seed: int = 20260701
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    attn_implementation: str | None = "sdpa"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def unique_temperatures(self) -> "RunSettings":
        if not self.temperatures or any(t <= 0 for t in self.temperatures):
            raise ValueError("temperatures must be non-empty and greater than zero")
        if len(set(self.temperatures)) != len(self.temperatures):
            raise ValueError("temperatures must be unique")
        return self


class JudgeSettings(StrictModel):
    enabled: bool = True
    model: str = "gemini-3-flash-preview"
    concurrency: int = Field(default=8, ge=1, le=64)
    max_attempts: int = Field(default=7, ge=1, le=20)
    request_timeout_seconds: int = Field(default=180, ge=10)
    temperature: float = Field(default=0.0, ge=0, le=2)


class ModelSpec(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    repo: str
    variant: Literal["base", "instruct"]
    loader: Literal["causal", "multimodal", "custom"] = "causal"
    gated: bool = False
    trust_remote_code: bool = False
    revision: str | None = None
    chat_template_kwargs: dict[str, bool] = Field(default_factory=dict)


class PromptSpec(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    category: Literal["factual", "creative", "reasoning"]
    title: str
    prompt: str
    reference_facts: list[str] = Field(default_factory=list)
    criteria: list[
        Literal[
            "coherence",
            "factuality",
            "creativity",
            "internal_consistency",
            "instruction_following",
            "fluency",
        ]
    ]
    constraints: dict[str, list[str]] = Field(default_factory=dict)


class BenchmarkConfig(StrictModel):
    run: RunSettings
    judge: JudgeSettings
    models_file: Path
    prompts_file: Path
    models: list[ModelSpec] = Field(default_factory=list)
    prompts: list[PromptSpec] = Field(default_factory=list)

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"models_file", "prompts_file"})
        # Operational controls may safely change while resuming. Semantic generation
        # and scoring settings remain in the fingerprint.
        payload["run"].pop("output_dir", None)
        payload["judge"].pop("concurrency", None)
        payload["judge"].pop("max_attempts", None)
        payload["judge"].pop("request_timeout_seconds", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    raw = _read_yaml(path)
    base = path.parent.parent if path.parent.name == "configs" else path.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    models_path = resolve(raw["models_file"])
    prompts_path = resolve(raw["prompts_file"])
    raw["models_file"] = models_path
    raw["prompts_file"] = prompts_path
    raw["models"] = _read_yaml(models_path)["models"]
    raw["prompts"] = _read_yaml(prompts_path)["prompts"]
    config = BenchmarkConfig.model_validate(raw)
    config.run.output_dir = resolve(str(config.run.output_dir))
    _ensure_unique("model", [item.id for item in config.models])
    _ensure_unique("prompt", [item.id for item in config.prompts])
    return config


def _ensure_unique(kind: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {kind} IDs: {', '.join(duplicates)}")
