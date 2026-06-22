"""
Tier-3 honeypot detector: cohort-relative outliers.

A candidate is suspicious if their numeric profile-shape features are
extreme compared to peers in the same (title_family, yoe_band) cohort.

We only penalize HIGH-SIDE outliers on features where being unusually
high is suspicious (e.g. claiming 30 skills when the cohort median is
12). Low-side outliers usually represent real specialists / juniors and
are not flagged.

Two-pass implementation:
  1. Walk the live pool to bucket candidates by cohort and accumulate
     per-cohort mean & std for each feature. Cohorts with < MIN_COHORT
     members fall back to global stats.
  2. Walk again to compute z-scores and accumulate a soft penalty into
     `ctx.components["honeypot_soft"]`. Capped so even an extreme
     candidate can't single-handedly dominate the penalty.

Like CrossFieldInconsistencyScorer, this stage never drops anyone.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ranker.heuristics import title_family
from ranker.pipeline import CandidateContext, Stage


# Features computed per candidate. Each is a function (candidate -> float)
# and a direction: only z-scores in this direction are penalized.
def _f_n_skills(c: dict[str, Any]) -> float:
    return float(len(c.get("skills") or []))


def _f_n_expert_skills(c: dict[str, Any]) -> float:
    return float(
        sum(
            1
            for s in c.get("skills") or []
            if (s.get("proficiency") or "").lower() in {"advanced", "expert"}
        )
    )


def _f_endorsements_sum(c: dict[str, Any]) -> float:
    return float(
        sum(int(s.get("endorsements") or 0) for s in c.get("skills") or [])
    )


def _f_skills_per_yoe(c: dict[str, Any]) -> float:
    yoe = (c.get("profile") or {}).get("years_of_experience") or 0
    n = len(c.get("skills") or [])
    return float(n / max(yoe, 0.5))  # avoid /0 for fresh grads


# Feature registry: feature_name -> (extractor, suspicious_direction)
# suspicious_direction is +1 if HIGH values are honeypot-y (the common case),
# or -1 if LOW values are. None means "either side is suspicious".
FEATURES: dict[str, tuple[callable, int]] = {
    "n_skills": (_f_n_skills, +1),
    "n_expert_skills": (_f_n_expert_skills, +1),
    "skills_per_yoe": (_f_skills_per_yoe, +1),
    "endorsements_sum": (_f_endorsements_sum, +1),
}


def _yoe_band(yoe: float | int | None) -> str:
    """Bucket years-of-experience into coarse bands matching the JD's framing."""
    if yoe is None:
        return "unknown"
    if yoe < 2:
        return "<2"
    if yoe < 5:
        return "2-5"
    if yoe < 10:
        return "5-10"
    return "10+"


def _cohort_key(c: dict[str, Any]) -> tuple[str, str]:
    profile = c.get("profile") or {}
    fam = title_family(profile.get("current_title")) or "unknown"
    band = _yoe_band(profile.get("years_of_experience"))
    return (fam, band)


class _RunningStats:
    """Welford's online mean/variance — avoids storing all values."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


class CohortOutlierScorer(Stage):
    name = "cohort_outlier"

    def __init__(
        self,
        z_threshold: float = 3.0,
        per_feature_penalty: float = 0.10,
        max_penalty: float = 0.30,
        min_cohort_size: int = 30,
    ):
        self.z_threshold = z_threshold
        self.per_feature_penalty = per_feature_penalty
        self.max_penalty = max_penalty
        self.min_cohort_size = min_cohort_size

    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        # ----- Pass 1: gather stats per (cohort, feature) -----------------
        cohort_stats: dict[tuple[tuple[str, str], str], _RunningStats] = defaultdict(_RunningStats)
        global_stats: dict[str, _RunningStats] = defaultdict(_RunningStats)
        cohort_sizes: dict[tuple[str, str], int] = defaultdict(int)

        # Cache features so pass 2 doesn't recompute.
        cache: dict[str, tuple[tuple[str, str], dict[str, float]]] = {}

        for ctx in pool:
            if ctx.is_dropped:
                continue
            cohort = _cohort_key(ctx.candidate)
            cohort_sizes[cohort] += 1
            feats: dict[str, float] = {}
            for fname, (fn, _) in FEATURES.items():
                v = fn(ctx.candidate)
                feats[fname] = v
                cohort_stats[(cohort, fname)].add(v)
                global_stats[fname].add(v)
            cache[ctx.candidate_id] = (cohort, feats)

        # ----- Pass 2: score each candidate -------------------------------
        flagged = 0
        feature_hits: dict[str, int] = {f: 0 for f in FEATURES}
        for ctx in pool:
            if ctx.is_dropped:
                continue
            cohort, feats = cache[ctx.candidate_id]
            cohort_big = cohort_sizes[cohort] >= self.min_cohort_size

            penalty = 0.0
            hit_descrs: list[str] = []
            for fname, (_, direction) in FEATURES.items():
                stats = (
                    cohort_stats[(cohort, fname)] if cohort_big else global_stats[fname]
                )
                if stats.n < 2 or stats.std <= 0:
                    continue
                z = (feats[fname] - stats.mean) / stats.std
                # Only count z in the suspicious direction.
                if direction == +1 and z < self.z_threshold:
                    continue
                if direction == -1 and z > -self.z_threshold:
                    continue
                if direction is None and abs(z) < self.z_threshold:
                    continue
                penalty += self.per_feature_penalty
                feature_hits[fname] += 1
                hit_descrs.append(f"{fname}(z={z:+.1f})")

            if penalty <= 0:
                continue
            penalty = min(penalty, self.max_penalty)
            ctx.components["honeypot_soft"] = (
                ctx.components.get("honeypot_soft", 0.0) + penalty
            )
            if penalty >= 0.2:
                ctx.notes.append(
                    f"Cohort outlier: {', '.join(hit_descrs)} (soft={penalty:.2f})"
                )
            flagged += 1

        print(f"    cohort_outlier: flagged {flagged:,} candidates")
        for fname, n in sorted(feature_hits.items(), key=lambda kv: -kv[1]):
            print(f"        {fname:<24} hit {n:>6,}")
        return pool
