from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Job:
    model_id: str
    prompt_id: str
    temperature: float
    sample: int
    seed: int
    sampling_profile: str = "unfiltered"

    @property
    def temperature_slug(self) -> str:
        return f"{self.temperature:g}".replace(".", "p")

    @property
    def key(self) -> str:
        profile = (
            ""
            if self.sampling_profile == "unfiltered"
            else f"__p{self.sampling_profile}"
        )
        return (
            f"{self.model_id}__{self.prompt_id}__t{self.temperature_slug}"
            f"{profile}__s{self.sample:02d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.generations = root / "generations"
        self.judgments = root / "judgments"
        self.failures = root / "failures"
        self.reports = root / "reports"

    def initialize(self, metadata: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.root / "run.json"
        if metadata_path.exists():
            current = read_json(metadata_path)
            if current.get("config_fingerprint") != metadata.get("config_fingerprint"):
                raise RuntimeError(
                    f"{self.root} contains results from a different configuration. "
                    "Use another --run-name or restore the original configuration."
                )
        else:
            atomic_write_json(metadata_path, metadata)

    def generation_path(self, job: Job) -> Path:
        return self.generations / job.model_id / f"{job.key}.json"

    def judgment_path(self, job: Job) -> Path:
        return self.judgments / job.model_id / f"{job.key}.json"

    def failure_path(self, stage: str, key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key)
        return self.failures / stage / f"{safe_key}.json"

    def iter_generations(self) -> Iterator[dict[str, Any]]:
        if not self.generations.exists():
            return
        for path in sorted(self.generations.rglob("*.json")):
            yield read_json(path)

    def iter_judgments(self) -> Iterator[dict[str, Any]]:
        if not self.judgments.exists():
            return
        for path in sorted(self.judgments.rglob("*.json")):
            yield read_json(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value
