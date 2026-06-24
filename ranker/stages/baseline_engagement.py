"""
BaselineEngagementScorer — a deliberately trivial placeholder scorer.

This exists so the pipeline can produce a valid submission CSV end-to-end
TODAY, before any real fit-scoring (BM25, embeddings, LLM rerank, etc.)
is wired in. Replace or stack additional scorer stages on top of this.

Scoring formula (intentionally simple, NOT good):
    score = 0.4 * profile_completeness/100
          + 0.3 * recruiter_response_rate
          + 0.2 * (1 if open_to_work else 0)
          + 0.1 * min(github_activity_score, 100)/100   (treat -1 as 0)

This privileges "available and engaged" candidates, which is at least
defensible per the JD ("a perfect-on-paper candidate who hasn't logged in
for 6 months ... is, for hiring purposes, not actually available").
"""

from __future__ import annotations

from ranker.pipeline import CandidateContext, Stage


class BaselineEngagementScorer(Stage):
    name = "baseline_engagement"

    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        for ctx in pool:
            if ctx.is_dropped:
                continue
            sig = ctx.candidate.get("redrob_signals") or {}

            pcs = (sig.get("profile_completeness_score") or 0) / 100.0
            rrr = sig.get("recruiter_response_rate") or 0.0
            otw = 1.0 if sig.get("open_to_work_flag") else 0.0
            gh_raw = sig.get("github_activity_score")
            gh = max(0.0, (gh_raw or 0.0)) / 100.0

            score = 0.4 * pcs + 0.3 * rrr + 0.2 * otw + 0.1 * gh

            # A short note for the reasoning column. Keep it factual and
            # specific to this candidate so we don't fail Stage 4 manual
            # review (templating / hallucination checks).
            yoe = (ctx.candidate.get("profile") or {}).get("years_of_experience")
            title = (ctx.candidate.get("profile") or {}).get("current_title")
            note = (
                f"Baseline engagement {score:.2f} "
                f"(complete={pcs:.0%}, response={rrr:.0%}, "
                f"{'open' if otw else 'not open'} to work); "
                f"{title} with {yoe} yrs experience."
            )
            ctx.add_score(self.name, score, note=note)
        return pool
