"""
IO helpers: streaming candidate loader and submission CSV writer.

We deliberately stream candidates one line at a time (the file is ~465 MB
uncompressed) and only materialize what we need.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from ranker.pipeline import CandidateContext

SUBMISSION_HEADER = ["candidate_id", "rank", "score", "reasoning"]
TOP_N = 100
SCORE_DECIMALS = 6  # precision used both for sorting and for CSV output


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate dicts from a .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"candidates file not found: {p}")
    opener = (
        (lambda: gzip.open(p, "rt", encoding="utf-8"))
        if p.suffix == ".gz"
        else (lambda: open(p, "r", encoding="utf-8"))
    )
    with opener() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"bad JSON on line {line_no}: {e}") from e


def build_pool(candidates_path: str | Path) -> list[CandidateContext]:
    return [CandidateContext(candidate=c) for c in iter_candidates(candidates_path)]


def _build_reasoning(ctx: CandidateContext, max_len: int = 280) -> str:
    """Stitch a candidate's accumulated notes into a single reasoning string."""
    if not ctx.notes:
        return ""
    text = " ".join(n.strip() for n in ctx.notes if n and n.strip())
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def write_submission(
    pool: list[CandidateContext],
    out_path: str | Path,
    top_n: int = TOP_N,
) -> list[CandidateContext]:
    """
    Pick the top-N live candidates and write a spec-compliant CSV.

    Sort order (matches the validator's requirements in submission_spec §3):
      1. score descending
      2. candidate_id ascending  (deterministic tiebreak)

    Returns the ranked top-N contexts (useful for inspection).
    """
    live = [c for c in pool if not c.is_dropped]
    if len(live) < top_n:
        raise RuntimeError(
            f"only {len(live)} live candidates remain after pipeline; need {top_n}"
        )

    # Round before sorting so the sort sees the same equality the
    # validator sees in the CSV (otherwise floats that print equal at
    # SCORE_DECIMALS but differ at higher precision can violate the
    # "ties broken by candidate_id ascending" rule).
    rounded: list[tuple[float, str, CandidateContext]] = [
        (round(c.score, SCORE_DECIMALS), c.candidate_id, c) for c in live
    ]
    rounded.sort(key=lambda t: (-t[0], t[1]))
    top = rounded[:top_n]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUBMISSION_HEADER)
        for rank, (rounded_score, _cid, ctx) in enumerate(top, start=1):
            w.writerow([
                ctx.candidate_id,
                rank,
                f"{rounded_score:.{SCORE_DECIMALS}f}",
                _build_reasoning(ctx),
            ])
    return [t[2] for t in top]
