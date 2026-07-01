from tempbench.config import PromptSpec
from tempbench.metrics import local_metrics


def test_repetition_and_constraints():
    prompt = PromptSpec(
        id="test",
        category="creative",
        title="Test",
        prompt="Test",
        criteria=["coherence"],
        constraints={
            "required_headings": ["I. Start"],
            "forbidden_phrases": ["just a dream"],
        },
    )
    result = local_metrics("I. Start\n\nred blue red blue red blue", prompt)
    assert result["word_count"] == 8
    assert result["repeated_bigram_ratio"] > 0
    assert result["all_machine_constraints_pass"] is True
