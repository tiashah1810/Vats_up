"""
Pipeline core: Stage abstraction, CandidateContext, and Pipeline runner.

Each Stage takes the full pool and returns it (possibly modified).
Stages communicate by mutating CandidateContext objects:
    - filter stages set .dropped_by / .drop_reason
    - scorer stages record per-component scores in .components and
      add to the cumulative .score
    - any stage can append a short human-readable line to .notes that
      will be used to build the submission's reasoning column.

This gives us a single, flexible primitive (Stage.run) that handles both
per-candidate transforms and pool-level operations (normalization,
re-ranking, deduplication, etc.) without growing a second abstraction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateContext:
    """Per-candidate state that flows through the pipeline."""

    candidate: dict[str, Any]
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dropped_by: str | None = None
    drop_reason: str | None = None

    @property
    def candidate_id(self) -> str:
        return self.candidate["candidate_id"]

    @property
    def is_dropped(self) -> bool:
        return self.dropped_by is not None

    def drop(self, stage_name: str, reason: str) -> None:
        """Mark this candidate as removed from the live pool."""
        self.dropped_by = stage_name
        self.drop_reason = reason

    def add_score(self, stage_name: str, value: float, note: str | None = None) -> None:
        """Record a scoring component and add it to the cumulative score."""
        self.components[stage_name] = value
        self.score += value
        if note:
            self.notes.append(note)


class Stage(ABC):
    """Base class for any pipeline stage.

    Subclasses set a class-level `name` and implement `run`. `run` receives
    the full pool and must return it (typically the same list, mutated in
    place). Stages should respect `ctx.is_dropped` and skip dropped
    candidates unless they explicitly need to inspect the full pool.
    """

    name: str = "stage"

    @abstractmethod
    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"


class Pipeline:
    """Runs a sequence of Stages and reports per-stage timing & pool size."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, pool: list[CandidateContext], verbose: bool = True) -> list[CandidateContext]:
        if verbose:
            print(f"pipeline start: {len(pool):,} candidates, {len(self.stages)} stages")
        for stage in self.stages:
            t0 = time.perf_counter()
            live_before = sum(1 for c in pool if not c.is_dropped)
            pool = stage.run(pool)
            live_after = sum(1 for c in pool if not c.is_dropped)
            dt = time.perf_counter() - t0
            if verbose:
                delta = live_after - live_before
                arrow = f"{live_before:,} -> {live_after:,}"
                tag = f"({delta:+,})" if delta else ""
                print(f"  [{stage.name:<24}] {arrow:>22} live  {tag:<8}  {dt:6.2f}s")
        if verbose:
            print(f"pipeline done: {sum(1 for c in pool if not c.is_dropped):,} live")
        return pool
