"""
HoneypotPenalty — final stage that applies the accumulated soft signal.

Reads `ctx.components["honeypot_soft"]` (filled by Tier-2 and Tier-3
detectors) and multiplies `ctx.score` by `(1 - clamp(soft, 0, cap))`.

This is the last stage in the pipeline because it has to see the final
fit-score before dampening it. A candidate with soft=0.5 keeps half
their score; a candidate with soft >= cap (default 0.9) is effectively
demoted to the bottom of the list without being dropped.
"""

from __future__ import annotations

from ranker.pipeline import CandidateContext, Stage


class HoneypotPenalty(Stage):
    name = "honeypot_penalty"

    def __init__(self, cap: float = 0.9, soft_component: str = "honeypot_soft"):
        if not 0 < cap <= 1.0:
            raise ValueError("cap must be in (0, 1]")
        self.cap = cap
        self.soft_component = soft_component

    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        n_applied = 0
        total_demotion = 0.0
        for ctx in pool:
            if ctx.is_dropped:
                continue
            soft = ctx.components.get(self.soft_component, 0.0)
            if soft <= 0:
                continue
            penalty_frac = min(soft, self.cap)
            old = ctx.score
            ctx.score *= 1.0 - penalty_frac
            # Track how much score we removed for inspection.
            ctx.components[self.name] = -(old - ctx.score)
            total_demotion += old - ctx.score
            n_applied += 1
        print(
            f"    honeypot_penalty: applied to {n_applied:,} candidates; "
            f"total score removed = {total_demotion:.2f}"
        )
        return pool
