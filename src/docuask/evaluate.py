"""Measure retrieval instead of guessing: a golden set of questions with the section that answers each one.

    python -m docuask.evaluate                      # print the table
    python -m docuask.evaluate --min-hit3 0.9       # exit 1 if hybrid hit@3 is below 0.9 (use it as a CI gate)

hit@1  the right section is the FIRST result      hit@3  it is among the first three
MRR    mean of 1/rank of the right section (1.0 = always first, 0 = never found)
The refusal check uses the nearest vector distance: positives should be below MAX_DISTANCE, negatives above it.
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session, sessionmaker

from docuask.config import Settings, get_settings
from docuask.db import get_session_factory
from docuask.providers import get_embeddings
from docuask.retrieval import HybridRetriever, search

MODES = ("vector", "keyword", "hybrid")


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_of_expected(docs, case: dict) -> int | None:
    for rank, d in enumerate(docs, start=1):
        if d.metadata["document"] == case["doc"] and case["section"] in (d.metadata["section"] or ""):
            return rank
    return None


def evaluate(golden: list[dict], sessions: sessionmaker[Session], embeddings: Embeddings, settings: Settings) -> dict:
    positives = [c for c in golden if c["expect"] == "answer"]
    negatives = [c for c in golden if c["expect"] == "refuse"]

    modes = {}
    for mode in MODES:
        retriever = HybridRetriever(
            sessions=sessions, embeddings=embeddings, mode=mode, k=10, candidate_k=settings.candidate_k
        )
        ranks = [rank_of_expected(retriever.invoke(c["q"]), c) for c in positives]
        n = len(ranks)
        modes[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "missed": [c["q"] for c, r in zip(positives, ranks, strict=True) if r is None or r > 3],
        }

    def nearest(case: dict) -> float:
        with sessions() as session:
            found = search(session, embeddings, case["q"], mode="vector", k=1, candidate_k=settings.candidate_k)
        return found.best_distance if found.best_distance is not None else 9.0

    pos_d = [nearest(c) for c in positives]
    neg_d = [nearest(c) for c in negatives]
    gate = {
        "max_distance": settings.max_distance,
        "positives_max": max(pos_d),
        "negatives_min": min(neg_d),
        "answered_ok": sum(d <= settings.max_distance for d in pos_d) / len(pos_d),
        "refused_ok": sum(d > settings.max_distance for d in neg_d) / len(neg_d),
    }
    return {"modes": modes, "gate": gate, "n_positive": len(positives), "n_negative": len(negatives)}


def to_markdown(result: dict) -> str:
    lines = [
        f"Questions: {result['n_positive']} answerable, {result['n_negative']} not covered by the documents.",
        "",
        "| search | hit@1 | hit@3 | MRR |",
        "|---|---|---|---|",
    ]
    for mode, m in result["modes"].items():
        lines.append(f"| {mode} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['mrr']:.2f} |")
    g = result["gate"]
    lines += [
        "",
        "Refusal gate (nearest vector distance):",
        "",
        f"- MAX_DISTANCE = {g['max_distance']}",
        f"- farthest answerable question: {g['positives_max']:.3f}; nearest unanswerable one: {g['negatives_min']:.3f}",
        f"- answerable questions let through: {g['answered_ok']:.0%}; unanswerable ones refused: {g['refused_ok']:.0%}",
    ]
    missed = result["modes"]["hybrid"]["missed"]
    if missed:
        lines += ["", "Hybrid search missed (not in the first 3):", *[f"- {q}" for q in missed]]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--report", default=None, help="also write the table to this file")
    parser.add_argument("--min-hit3", type=float, default=None)
    args = parser.parse_args()

    result = evaluate(load_golden(Path(args.golden)), get_session_factory(), get_embeddings(), get_settings())
    text = to_markdown(result)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    if args.min_hit3 is not None and result["modes"]["hybrid"]["hit@3"] < args.min_hit3:
        print(
            f"\nFAIL: hybrid hit@3 {result['modes']['hybrid']['hit@3']:.0%} is below {args.min_hit3:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
