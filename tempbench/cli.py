from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from tempbench.config import BenchmarkConfig, ModelSpec, PromptSpec, load_config
from tempbench.environment import configure_huggingface_cache
from tempbench.jobs import create_jobs, select_models, select_prompts
from tempbench.storage import Job, RunStore, atomic_write_json

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tempbench",
        description="Benchmark local model degradation as sampling temperature rises.",
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--run-name", default="main")
    parser.add_argument(
        "--models",
        action="append",
        help="Model ID or comma-separated IDs. May be repeated. Defaults to all.",
    )
    parser.add_argument(
        "--prompts",
        action="append",
        help="Prompt ID or comma-separated IDs. May be repeated. Defaults to all.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Print the job matrix without loading models.")
    preflight = subparsers.add_parser("preflight", help="Check environment and model access.")
    preflight.add_argument(
        "--remote", action="store_true", help="Also query Hugging Face for every checkpoint."
    )
    subparsers.add_parser("generate", help="Run only local GPU generation.")
    subparsers.add_parser("judge", help="Judge existing generations with Gemini.")
    subparsers.add_parser("run", help="Generate locally while judging concurrently.")
    subparsers.add_parser("report", help="Rebuild CSV and HTML reports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    cache_path = configure_huggingface_cache()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        models = select_models(config, _split_filters(args.models))
        prompts = select_prompts(config, _split_filters(args.prompts))
        store = RunStore(config.run.output_dir / args.run_name)
        if args.command == "plan":
            return command_plan(config, models, prompts, store)
        if args.command == "preflight":
            console.print(f"Hugging Face cache: {cache_path}")
            return command_preflight(config, models, remote=args.remote)
        if args.command == "report":
            return command_report(store)
        store.initialize(_run_metadata(config, args.run_name))
        if args.command == "generate":
            return command_generate(config, models, prompts, store, args.fail_fast, None)
        if args.command == "judge":
            return command_judge(config, models, prompts, store)
        if args.command == "run":
            return command_run(config, models, prompts, store, args.fail_fast)
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/yellow] Completed artifacts are safe; rerun to resume."
        )
        return 130
    except Exception as exc:
        console.print(f"[red]Fatal error:[/red] {exc}")
        if os.getenv("TEMPBENCH_DEBUG"):
            traceback.print_exc()
        return 1
    return 0


def _split_filters(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def command_plan(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
    store: RunStore,
) -> int:
    jobs = create_jobs(config, models, prompts)
    complete = sum(store.generation_path(job).exists() for job in jobs)
    estimated_words = len(jobs) * 725
    table = Table(title="Benchmark plan")
    table.add_column("Item")
    table.add_column("Value", justify="right")
    table.add_row("Models", str(len(models)))
    table.add_row("Prompts", str(len(prompts)))
    table.add_row("Temperatures", str(len(config.run.temperatures)))
    table.add_row("Samples / condition", str(config.run.samples_per_condition))
    table.add_row("Total generations", str(len(jobs)))
    table.add_row("Already generated", str(complete))
    table.add_row("Approx. output words", f"{estimated_words:,}")
    table.add_row("Result directory", str(store.root))
    console.print(table)
    console.print("\nModels: " + ", ".join(model.id for model in models))
    return 0


def command_preflight(config: BenchmarkConfig, models: list[ModelSpec], *, remote: bool) -> int:
    failures = 0
    console.print(f"Python {platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 11):
        console.print("[red]FAIL[/red] Python 3.11 or newer is required")
        failures += 1
    try:
        import torch
        import transformers

        console.print(f"PyTorch {torch.__version__}; Transformers {transformers.__version__}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gib = props.total_memory / 1024**3
                console.print(f"[green]GPU {index}[/green] {props.name}, {gib:.1f} GiB")
            if config.run.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                console.print("[red]FAIL[/red] Config requests bfloat16 but GPU lacks BF16 support")
                failures += 1
        else:
            console.print("[red]FAIL[/red] CUDA is not available to PyTorch")
            failures += 1
    except ImportError as exc:
        console.print(f"[red]FAIL[/red] GPU dependencies are not installed: {exc}")
        failures += 1

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if config.judge.enabled and not gemini_key:
        console.print("[red]FAIL[/red] GEMINI_API_KEY is absent")
        failures += 1
    else:
        console.print("[green]OK[/green] Gemini API key is present")

    if any(model.gated for model in models) and not (
        os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    ):
        console.print("[red]FAIL[/red] HF_TOKEN is required for gated Llama/Gemma models")
        failures += 1
    else:
        console.print("[green]OK[/green] Hugging Face token requirement")

    if remote:
        failures += _check_remote_models(models)
        if config.judge.enabled and gemini_key:
            failures += _check_gemini(config, gemini_key)
    if failures:
        console.print(f"[red]Preflight failed with {failures} problem(s).[/red]")
        return 1
    console.print("[green]Preflight passed.[/green]")
    return 0


def _check_remote_models(models: list[ModelSpec]) -> int:
    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    failures = 0
    for model in models:
        try:
            info = api.model_info(model.repo, revision=model.revision, timeout=30)
            console.print(f"[green]Accessible[/green] {model.repo} @ {info.sha[:10]}")
        except Exception as exc:
            failures += 1
            console.print(f"[red]Inaccessible[/red] {model.repo}: {exc}")
    return failures


def _check_gemini(config: BenchmarkConfig, api_key: str) -> int:
    from google import genai

    try:
        client = genai.Client(api_key=api_key)
        info = client.models.get(model=config.judge.model)
        console.print(
            f"[green]Accessible[/green] Gemini judge model "
            f"{getattr(info, 'name', config.judge.model)}"
        )
        return 0
    except Exception as exc:
        console.print(f"[red]Inaccessible[/red] Gemini judge model: {exc}")
        return 1


def command_run(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
    store: RunStore,
    fail_fast: bool,
) -> int:
    judge_pool = _start_judge_pool(config, models, prompts, store)
    generation_code = 0
    judge_failures = 0
    try:
        generation_code = command_generate(config, models, prompts, store, fail_fast, judge_pool)
    finally:
        if judge_pool is not None:
            console.print(
                f"[cyan]Generation phase complete.[/cyan] Waiting for "
                f"{len(judge_pool.futures)} Gemini judgment(s)."
            )
            _, judge_failures = judge_pool.drain()
    command_report(store)
    return 1 if generation_code or judge_failures else 0


def command_generate(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
    store: RunStore,
    fail_fast: bool,
    judge_pool,
) -> int:
    from tempbench.generate import TransformersGenerator

    jobs = create_jobs(config, models, prompts)
    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    jobs_by_model: dict[str, list[Job]] = defaultdict(list)
    for job in jobs:
        if not store.generation_path(job).exists():
            jobs_by_model[job.model_id].append(job)
    pending = sum(map(len, jobs_by_model.values()))
    console.print(f"[cyan]Generation[/cyan] {pending} pending of {len(jobs)} selected jobs")
    failures = 0
    completed = len(jobs) - pending

    def save_result(job: Job, prompt: PromptSpec, record: dict[str, Any]) -> None:
        nonlocal completed
        atomic_write_json(store.generation_path(job), record)
        completed += 1
        console.print(
            f"[green]Generated[/green] {job.key} "
            f"({completed}/{len(jobs)}, {record['local_metrics']['word_count']} words)"
        )
        if judge_pool is not None:
            judge_pool.submit(job, prompt, record)

    model_by_id = {model.id: model for model in models}
    for model_id, model_jobs in jobs_by_model.items():
        model = model_by_id[model_id]
        try:
            with TransformersGenerator(model, config.run) as generator:
                batch_size = 1 if model.loader == "custom" else config.run.batch_size
                if model.loader == "custom" and config.run.batch_size != 1:
                    console.print(
                        f"[yellow]{model.id}[/yellow] uses batch_size=1 because its "
                        "custom AR method does not accept an attention mask."
                    )
                grouped: dict[float, list[Job]] = defaultdict(list)
                for job in model_jobs:
                    grouped[job.temperature].append(job)
                for temperature in config.run.temperatures:
                    temperature_jobs = grouped.get(temperature, [])
                    for start in range(0, len(temperature_jobs), batch_size):
                        batch_jobs = temperature_jobs[start : start + batch_size]
                        items = [(job, prompt_by_id[job.prompt_id]) for job in batch_jobs]
                        try:
                            generator.generate_batch(items, save_result)
                        except Exception as exc:
                            failures += len(items)
                            for job, _prompt in items:
                                _write_failure(store, "generate", job, exc)
                            console.print(
                                f"[red]Generation failed[/red] {model_id}, "
                                f"T={temperature:g}, batch={len(items)}: {exc}"
                            )
                            if fail_fast:
                                raise
        except Exception as exc:
            remaining = [job for job in model_jobs if not store.generation_path(job).exists()]
            failures += len(remaining)
            for job in remaining:
                _write_failure(store, "model", job, exc)
            console.print(f"[red]Model failed[/red] {model_id}: {exc}")
            if fail_fast:
                raise
    console.print(f"Generation complete: {completed} present, {failures} failure event(s).")
    return 1 if failures else 0


def command_judge(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
    store: RunStore,
) -> int:
    pool = _start_judge_pool(config, models, prompts, store)
    if pool is None:
        console.print("[yellow]Judging is disabled in the configuration.[/yellow]")
        return 0
    console.print(f"Waiting for {len(pool.futures)} Gemini judgment(s).")
    _, failures = pool.drain()
    command_report(store)
    return 1 if failures else 0


def _start_judge_pool(
    config: BenchmarkConfig,
    models: list[ModelSpec],
    prompts: list[PromptSpec],
    store: RunStore,
):
    if not config.judge.enabled:
        return None
    from tempbench.judge import GeminiJudgePool, job_from_generation

    prompt_by_id = {prompt.id: prompt for prompt in prompts}
    model_ids = {model.id for model in models}
    pool = GeminiJudgePool(config.judge, store)
    submitted = 0
    for generation in store.iter_generations():
        job = job_from_generation(generation)
        prompt = prompt_by_id.get(job.prompt_id)
        if (
            job.model_id in model_ids
            and prompt is not None
            and pool.submit(job, prompt, generation)
        ):
            submitted += 1
    console.print(
        f"[cyan]Gemini judge pool[/cyan] concurrency={config.judge.concurrency}, "
        f"queued existing={submitted}"
    )
    return pool


def command_report(store: RunStore) -> int:
    from tempbench.report import build_reports

    paths = build_reports(store)
    console.print(
        "[green]Reports updated:[/green] "
        + ", ".join(f"{name}={path}" for name, path in paths.items())
    )
    return 0


def _write_failure(store: RunStore, stage: str, job: Job, exc: Exception) -> None:
    atomic_write_json(
        store.failure_path(stage, job.key),
        {
            "job": job.to_dict(),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": datetime.now(UTC).isoformat(),
        },
    )


def _run_metadata(config: BenchmarkConfig, run_name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": run_name,
        "config_fingerprint": config.fingerprint(),
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "host": platform.node(),
        "temperatures": config.run.temperatures,
        "samples_per_condition": config.run.samples_per_condition,
    }


if __name__ == "__main__":
    raise SystemExit(main())
