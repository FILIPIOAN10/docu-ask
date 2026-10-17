from pathlib import Path

from docuask.evaluate import evaluate as run_evaluation
from docuask.evaluate import load_golden, to_markdown

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden.jsonl"


def test_hybrid_search_finds_the_right_section_for_nearly_every_golden_question(loaded, embeddings, settings):
    result = run_evaluation(load_golden(GOLDEN), loaded, embeddings, settings)
    hybrid = result["modes"]["hybrid"]
    assert hybrid["hit@3"] >= 0.9
    assert hybrid["hit@1"] >= max(
        result["modes"]["vector"]["hit@1"], result["modes"]["keyword"]["hit@1"]
    )  # fusion helps
    assert result["n_positive"] == 22 and result["n_negative"] == 6


def test_the_report_is_a_markdown_table(loaded, embeddings, settings):
    text = to_markdown(run_evaluation(load_golden(GOLDEN), loaded, embeddings, settings))
    assert "| hybrid |" in text and "MAX_DISTANCE" in text
