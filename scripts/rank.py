#!/usr/bin/env python3
"""
Build the current top-100 submission by running the ranking pipeline.

Usage:
    uv run python scripts/rank.py \
        --candidates data/India_runs_data_and_ai_challenge/candidates.jsonl \
        --honeypots  artifacts/honeypots.csv \
        --out        artifacts/submission.csv

Prerequisite: run `scripts/find_honeypots.py --hard-only` first to
produce the honeypot CSV the filter consumes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the repo root importable when running as a plain script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranker import Pipeline, write_submission
from ranker.io import build_pool
from ranker.stages import (
    BaselineEngagementScorer,
    CohortOutlierScorer,
    CrossFieldInconsistencyScorer,
    HoneypotFilter,
    HoneypotPenalty,
)


def build_pipeline(honeypots_csv: Path) -> Pipeline:
    """Assemble the ordered list of stages.

    Pipeline structure (current):
      Tier-1 (hard drop):  HoneypotFilter
      Fit scoring:         BaselineEngagementScorer  (placeholder; real
                                                      JD-fit scorer goes here)
      Tier-2 (soft):       CrossFieldInconsistencyScorer
      Tier-3 (soft):       CohortOutlierScorer
      Score dampener:      HoneypotPenalty  (multiplies score by 1 - soft)

    Add new stages here as the ranker grows (e.g. JD keyword match,
    embedding similarity, LLM rerank, behavioral-signal multiplier).
    HoneypotPenalty must remain last so it sees the final fit score.
    """
    return Pipeline([
        HoneypotFilter.from_csv(honeypots_csv, severity="HARD"),
        BaselineEngagementScorer(),
        CrossFieldInconsistencyScorer(),
        CohortOutlierScorer(),
        HoneypotPenalty(cap=0.9),
    ])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        default="data/India_runs_data_and_ai_challenge/candidates.jsonl",
    )
    p.add_argument("--honeypots", default="artifacts/honeypots.csv")
    p.add_argument("--out", default="artifacts/submission.csv")
    p.add_argument("--top-n", type=int, default=100)
    args = p.parse_args()

    t0 = time.perf_counter()
    print(f"loading candidates from {args.candidates} ...")
    pool = build_pool(args.candidates)
    print(f"loaded {len(pool):,} in {time.perf_counter() - t0:.2f}s")

    pipeline = build_pipeline(Path(args.honeypots))
    pool = pipeline.run(pool)

    top = write_submission(pool, args.out, top_n=args.top_n)

    # Quick sanity report — any honeypot in the top-100 is a DQ.
    print(f"\nwrote {len(top)} rows -> {args.out}")
    print(f"top score = {top[0].score:.4f}   bottom score = {top[-1].score:.4f}")
    print(f"total runtime: {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
