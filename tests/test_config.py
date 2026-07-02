from pathlib import Path

from tempbench.cli import _split_filters, build_parser
from tempbench.config import load_config
from tempbench.jobs import create_jobs, select_models, select_prompts


ROOT = Path(__file__).parents[1]


def test_default_matrix_has_expected_size():
    config = load_config(ROOT / "configs/benchmark.yaml")
    selected = select_models(config, None)
    jobs = create_jobs(config, selected, config.prompts)
    assert len(config.models) == 18
    assert len(selected) == 16
    assert len(config.prompts) == 6
    assert len(jobs) == 16 * 6 * 6
    assert {job.temperature for job in jobs} == {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}


def test_temperature_conditions_use_paired_seed():
    config = load_config(ROOT / "configs/benchmark.yaml")
    jobs = create_jobs(config, config.models[:1], config.prompts[:1])
    assert len({job.seed for job in jobs}) == 1


def test_sampler_pilot_matrix_and_paired_seeds():
    config = load_config(ROOT / "configs/sampler-pilot.yaml")
    models = select_models(
        config,
        [
            "llama-3.1-8b-base",
            "llama-3.1-8b-instruct",
            "qwen-2.5-14b-base",
            "qwen-2.5-14b-instruct",
        ],
    )
    prompts = select_prompts(
        config,
        ["everest_facts", "causal_clockwork_story", "contradiction_repair"],
    )
    jobs = create_jobs(config, models, prompts)
    assert len(jobs) == 4 * 3 * 4 * 6
    assert {job.sampling_profile for job in jobs} == {
        "unfiltered",
        "top-k-20",
        "top-k-60",
        "top-p-090",
        "min-p-005",
        "combined",
    }
    paired = [
        job
        for job in jobs
        if job.model_id == "llama-3.1-8b-base" and job.prompt_id == "everest_facts"
    ]
    assert len({job.seed for job in paired}) == 1


def test_cli_filters_do_not_consume_subcommand():
    args = build_parser().parse_args(
        [
            "--models",
            "model-a,model-b",
            "--prompts",
            "prompt-a",
            "--prompts",
            "prompt-b",
            "plan",
        ]
    )
    assert args.command == "plan"
    assert _split_filters(args.models) == ["model-a", "model-b"]
    assert _split_filters(args.prompts) == ["prompt-a", "prompt-b"]
