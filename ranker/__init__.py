"""
Pipeline framework for the Redrob ranking challenge.

Public API:
    Pipeline    -- composes Stages and runs them on a candidate pool
    Stage       -- ABC: each stage transforms the pool
    CandidateContext -- per-candidate state carried through stages
    iter_candidates, write_submission -- IO helpers
"""

from ranker.pipeline import CandidateContext, Pipeline, Stage
from ranker.io import iter_candidates, write_submission

__all__ = [
    "CandidateContext",
    "Pipeline",
    "Stage",
    "iter_candidates",
    "write_submission",
]
