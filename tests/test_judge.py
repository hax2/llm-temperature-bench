from concurrent.futures import Future

from tempbench.config import JudgeSettings
from tempbench.judge import GeminiJudgePool
from tempbench.storage import Job, RunStore


def test_completed_judgment_is_persisted_before_drain(tmp_path):
    store = RunStore(tmp_path / "run")
    pool = GeminiJudgePool(JudgeSettings(), store)
    job = Job("model", "prompt", 2.0, 1, 42, "top-k-20")
    future = Future()
    future.set_result(
        {
            "job": job.to_dict(),
            "scores": {"overall": {"score": 7, "rationale": "Coherent."}},
        }
    )

    assert pool._persist_completed(future, job) is True
    assert store.judgment_path(job).is_file()
    pool.executor.shutdown(wait=True)
