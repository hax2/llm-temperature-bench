from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from tempbench.config import JudgeSettings, PromptSpec
from tempbench.storage import Job, RunStore, atomic_write_json

console = Console()


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    rationale: str


class ClaimAssessment(BaseModel):
    claim: str
    verdict: str = Field(description="supported, contradicted, unverifiable, or not_applicable")
    explanation: str


class JudgeResponse(BaseModel):
    coherence: CriterionScore
    internal_consistency: CriterionScore
    instruction_following: CriterionScore
    fluency: CriterionScore
    factuality: CriterionScore | None = None
    creativity: CriterionScore | None = None
    overall: CriterionScore
    factual_claims: list[ClaimAssessment] = Field(default_factory=list)
    major_problems: list[str] = Field(default_factory=list)
    summary: str


class GeminiJudgePool:
    def __init__(self, settings: JudgeSettings, store: RunStore):
        self.settings = settings
        self.store = store
        self.executor = ThreadPoolExecutor(
            max_workers=settings.concurrency, thread_name_prefix="gemini-judge"
        )
        self.futures: dict[Future, Job] = {}
        self.local = threading.local()
        self.persist_lock = threading.Lock()
        self.outcomes: dict[str, bool] = {}

    def submit(
        self,
        job: Job,
        prompt: PromptSpec,
        generation: dict[str, Any],
        *,
        force: bool = False,
    ) -> bool:
        target = self.store.judgment_path(job)
        if target.exists() and not force:
            return False
        future = self.executor.submit(self._judge_one, job, prompt, generation)
        self.futures[future] = job
        future.add_done_callback(
            lambda completed, submitted_job=job: self._persist_completed(
                completed, submitted_job
            )
        )
        return True

    def drain(self) -> tuple[int, int]:
        succeeded = 0
        failed = 0
        total = len(self.futures)
        for index, future in enumerate(as_completed(self.futures), 1):
            job = self.futures[future]
            if self._persist_completed(future, job):
                succeeded += 1
            else:
                failed += 1
            console.print(f"[cyan]Judge drain[/cyan] {index}/{total}")
        self.futures.clear()
        self.executor.shutdown(wait=True)
        return succeeded, failed

    def _persist_completed(self, future: Future, job: Job) -> bool:
        with self.persist_lock:
            previous = self.outcomes.get(job.key)
            if previous is not None:
                return previous
            try:
                record = future.result()
                atomic_write_json(self.store.judgment_path(job), record)
                outcome = True
                console.print(
                    f"[green]Judged[/green] {job.key} "
                    f"(overall={record['scores']['overall']['score']})"
                )
            except Exception as exc:
                outcome = False
                atomic_write_json(
                    self.store.failure_path("judge", job.key),
                    {
                        "job": job.to_dict(),
                        "stage": "judge",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_at": datetime.now(UTC).isoformat(),
                    },
                )
                console.print(f"[red]Judge failed[/red] {job.key}: {exc}")
            self.outcomes[job.key] = outcome
            return outcome

    def _judge_one(
        self, job: Job, prompt: PromptSpec, generation: dict[str, Any]
    ) -> dict[str, Any]:
        client = self._client()
        request = _build_request(prompt, generation["output"])
        started = time.monotonic()
        response = None
        retryer = Retrying(
            stop=stop_after_attempt(self.settings.max_attempts),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                from google.genai import types

                response = client.models.generate_content(
                    model=self.settings.model,
                    contents=request,
                    config=types.GenerateContentConfig(
                        temperature=self.settings.temperature,
                        response_mime_type="application/json",
                        response_schema=JudgeResponse,
                    ),
                )
                if response.parsed is None:
                    if not response.text:
                        raise RuntimeError("Gemini returned neither parsed data nor text")
                    parsed = JudgeResponse.model_validate_json(response.text)
                elif isinstance(response.parsed, JudgeResponse):
                    parsed = response.parsed
                else:
                    parsed = JudgeResponse.model_validate(response.parsed)

        score_data = parsed.model_dump()
        usage = getattr(response, "usage_metadata", None)
        return {
            "schema_version": 1,
            "job": job.to_dict(),
            "judge": {
                "provider": "google",
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "blind_to_model_and_temperature": True,
            },
            "scores": score_data,
            "usage": _serializable_usage(usage),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "judged_at": datetime.now(UTC).isoformat(),
        }

    def _client(self):
        if not hasattr(self.local, "client"):
            from google import genai
            from google.genai import types

            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY in .env before judging")
            self.local.client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=self.settings.request_timeout_seconds * 1000
                ),
            )
        return self.local.client


def _build_request(prompt: PromptSpec, output: str) -> str:
    applicable = ", ".join(prompt.criteria)
    facts = "\n".join(f"- {fact}" for fact in prompt.reference_facts) or "(none supplied)"
    return f"""You are evaluating one anonymous model response in a controlled benchmark.
The model identity and sampling temperature are deliberately hidden. Do not infer or discuss them.

Score every applicable dimension from 1 (unusable) to 10 (excellent). Use the entire scale.
Coherence means global logical flow, causal continuity, organization, and absence of degeneration.
Internal consistency means names, dates, claims, chronology, and premises do not conflict.
Instruction following includes all explicit format, length, content, and constraint requirements.
Fluency concerns readable grammar and phrasing, separate from factual accuracy.
Factuality applies to factual and reasoning tasks. Penalize invented specifics and unsupported
claims. The reference sheet is authoritative but not exhaustive.
Creativity applies only to creative tasks; novelty must not compensate for incoherence.
Overall should reflect the applicable dimensions, not a mechanical average.

Applicable criteria: {applicable}
Task category: {prompt.category}
Task:
{prompt.prompt}

Reference sheet:
{facts}

Anonymous response:
--- BEGIN RESPONSE ---
{output}
--- END RESPONSE ---

Return the requested structured assessment. For non-applicable factuality or creativity, return null.
List concrete major problems. In factual_claims, include material errors or questionable claims,
not every correct sentence. Keep each rationale concise and evidence-based."""


def _serializable_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    try:
        return json.loads(json.dumps(usage))
    except TypeError:
        return {"repr": repr(usage)}


def job_from_generation(record: dict[str, Any]) -> Job:
    return Job(**record["job"])
