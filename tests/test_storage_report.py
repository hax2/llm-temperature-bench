import csv

from tempbench.report import build_reports
from tempbench.storage import Job, RunStore, atomic_write_json


def test_report_builds_from_partial_results(tmp_path):
    store = RunStore(tmp_path / "run")
    job = Job("model", "prompt", 1.5, 1, 42)
    generation = {
        "job": job.to_dict(),
        "model": {"repo": "org/model", "variant": "base"},
        "prompt": {"category": "factual"},
        "output": "Example output.",
        "local_metrics": {
            "word_count": 2,
            "unique_word_ratio": 1.0,
            "repeated_bigram_ratio": 0.0,
            "repeated_trigram_ratio": 0.0,
            "all_machine_constraints_pass": None,
        },
    }
    atomic_write_json(store.generation_path(job), generation)
    paths = build_reports(store)
    assert paths["html"].is_file()
    with paths["results_csv"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["judged"] == "False"
    assert rows[0]["sampling_profile"] == "unfiltered"
    assert rows[0]["temperature"] == "1.5"


def test_sampling_profile_is_part_of_job_identity():
    original = Job("model", "prompt", 2.0, 1, 42)
    filtered = Job("model", "prompt", 2.0, 1, 42, "top-k-20")
    assert original.key == "model__prompt__t2__s01"
    assert filtered.key == "model__prompt__t2__ptop-k-20__s01"
