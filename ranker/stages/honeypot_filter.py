"""
HoneypotFilter stage.

Removes candidates listed in a honeypot CSV (produced by
`scripts/find_honeypots.py`). Honeypots are profiles with internal
impossibilities; including them in the top 100 risks a >10% honeypot
rate which is an automatic disqualifier (submission_spec §7).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ranker.pipeline import CandidateContext, Stage


class HoneypotFilter(Stage):
    name = "honeypot_filter"

    def __init__(self, honeypot_ids: set[str], require_severity: str | None = "HARD"):
        """
        Args:
            honeypot_ids: set of candidate_id strings to drop.
            require_severity: kept for documentation; filtering already
                happened when the CSV was built (use --hard-only on
                find_honeypots.py to restrict to HARD).
        """
        self.honeypot_ids = honeypot_ids
        self.require_severity = require_severity

    @classmethod
    def from_csv(cls, path: str | Path, severity: str | None = "HARD") -> "HoneypotFilter":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"honeypot CSV not found: {p}. "
                f"Run: uv run python scripts/find_honeypots.py --hard-only"
            )
        ids: set[str] = set()
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if severity and row.get("severity") and row["severity"] != severity:
                    continue
                cid = (row.get("candidate_id") or "").strip()
                if cid:
                    ids.add(cid)
        return cls(ids, require_severity=severity)

    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        if not self.honeypot_ids:
            return pool
        dropped = 0
        for ctx in pool:
            if ctx.is_dropped:
                continue
            if ctx.candidate_id in self.honeypot_ids:
                ctx.drop(self.name, "flagged as honeypot (impossible profile)")
                dropped += 1
        # Loud-and-proud trace: dropping the wrong candidates here is silent
        # death for the submission.
        print(f"    honeypot_filter: dropped {dropped} of {len(self.honeypot_ids)} known honeypots")
        return pool
