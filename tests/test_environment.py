import os

from tempbench.environment import configure_huggingface_cache


def test_explicit_hf_home_wins(tmp_path, monkeypatch):
    target = tmp_path / "models"
    monkeypatch.setenv("HF_HOME", str(target))
    assert configure_huggingface_cache() == target
    assert target.is_dir()
    assert os.environ["HF_HOME"] == str(target)
