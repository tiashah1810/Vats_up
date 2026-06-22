"""
Tier-2 honeypot detector: cross-field inconsistencies.

Honeypots in this dataset are synthesized by stitching independently
sampled fields together. So the same fact, asserted in two places, will
disagree more often for fakes than for real candidates.

Each rule is a soft signal — a single hit is suggestive, not conclusive.
We accumulate weighted hits into `ctx.components["honeypot_soft"]` which
is later turned into a multiplicative score penalty by HoneypotPenalty.
We DO NOT drop anyone here; only the Tier-1 HoneypotFilter has that power.

Rules included (and why some are currently disabled, see ENABLED_RULES):

  * assessment_contradicts_proficiency .. expert/advanced skill but
                                          assessment <30. High-precision
                                          (~3% hit rate); kept ON.
  * unrelated_skill_cluster ............. skills span >= 6 unrelated
                                          domains. Real polymaths exist
                                          but >=6 is rare; kept ON.
  * too_many_skills_for_yoe ............. > 3 * yoe + 5 skills listed.
                                          ~6% hit rate; kept ON.

  * industry_mismatch / company_size_mismatch ............... OFF.
        Both produced 0 hits on the full 100k pool — the synthetic
        generator copies current job's industry/size into the profile.
        Dead signals in this dataset.

  * title_vs_description ................................... OFF.
        Naive token-bag matching produced ~49% hit rate; needs real
        semantic matching (embeddings) to be precise. Re-enable after
        a better classifier is wired in.

  * summary_vs_title ....................................... OFF.
        Same problem — ~29% hit rate from a fragile regex. Compiled
        patterns are kept module-level for the day this gets revived.
"""

from __future__ import annotations

import re
from typing import Any

from ranker.heuristics import FAMILY_DESC_TOKENS, TITLE_TOKENS, title_family
from ranker.pipeline import CandidateContext, Stage


# Active rules. Set to False to disable; the rule code stays present so
# we can revive it after improving its heuristic.
ENABLED_RULES: dict[str, bool] = {
    "assessment_contradicts_proficiency": True,
    "unrelated_skill_cluster": True,
    "too_many_skills_for_yoe": True,
    "industry_mismatch": False,
    "company_size_mismatch": False,
    "title_vs_description": False,
    "summary_vs_title": False,
}


# Skills grouped into broad domains. Used by `unrelated_skill_cluster`
# to count how many disjoint specialties a candidate claims.
SKILL_BUCKETS: dict[str, set[str]] = {
    "frontend": {
        "react", "vue", "angular", "next.js", "typescript", "javascript",
        "tailwind", "webpack", "html", "css",
    },
    "backend": {
        "node.js", "go", "rust", "java", "spring", "django", "flask",
        "grpc", "rest", "fastapi", "php", "ruby on rails",
    },
    "data_eng": {
        "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake",
        "bigquery", "etl", "sql", "apache beam",
    },
    "ml": {
        "pytorch", "tensorflow", "scikit-learn", "xgboost", "lora",
        "fine-tuning llms", "nlp", "gans", "yolo", "diffusion models",
        "speech recognition", "tts", "image classification",
    },
    "infra": {
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "ansible", "jenkins",
    },
    "design": {"figma", "photoshop", "illustrator", "sketch"},
    "enterprise": {
        "sap", "salesforce crm", "tally", "excel", "six sigma",
        "accounting", "powerpoint",
    },
    "vectordb": {"pinecone", "weaviate", "qdrant", "milvus", "faiss"},
    "mobile": {"swift", "kotlin", "react native", "flutter"},
}

# Threshold for the unrelated-skill-cluster rule. 5 was too lenient
# (real polymaths fire); 6 cuts the hit rate sharply.
SKILL_CLUSTER_THRESHOLD = 6


# Pre-compile the summary-vs-title patterns once at module load. Even
# though that rule is currently disabled, leaving the compiled patterns
# here makes re-enabling it a one-line change.
_SUMMARY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    fam: [
        re.compile(
            rf"(?:background (?:is )?in|my professional background is in|"
            rf"i'?m an? )\s*[^.]{{0,40}}\b{re.escape(tok)}\b"
        )
        for tok in tokens
    ]
    for fam, tokens in TITLE_TOKENS.items()
}


def _description_family_hits(description: str) -> dict[str, int]:
    """Count token-bag hits per family in a description string."""
    text = (description or "").lower()
    if not text:
        return {}
    out: dict[str, int] = {}
    for fam, tokens in FAMILY_DESC_TOKENS.items():
        n = sum(1 for tok in tokens if tok in text)
        if n:
            out[fam] = n
    return out


class CrossFieldInconsistencyScorer(Stage):
    """Tier-2 soft scorer; only writes to components['honeypot_soft']."""

    name = "cross_field_check"

    DEFAULT_WEIGHTS: dict[str, float] = {
        # Self-contradiction within signals: high confidence.
        "assessment_contradicts_proficiency": 0.40,
        # Profile-shape outliers: moderate (real polymaths exist).
        "unrelated_skill_cluster": 0.25,
        "too_many_skills_for_yoe": 0.15,
        # Disabled rules (kept here so weights are documented):
        "industry_mismatch": 0.25,
        "company_size_mismatch": 0.15,
        "title_vs_description": 0.30,
        "summary_vs_title": 0.20,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        enabled: dict[str, bool] | None = None,
    ):
        self.weights = dict(weights) if weights else dict(self.DEFAULT_WEIGHTS)
        self.enabled = dict(enabled) if enabled else dict(ENABLED_RULES)

    # --- individual rule predicates ---------------------------------------

    def _rule_industry_mismatch(self, c: dict[str, Any]) -> bool:
        p_ind = (c.get("profile", {}).get("current_industry") or "").strip().lower()
        for j in c.get("career_history", []) or []:
            if j.get("is_current"):
                j_ind = (j.get("industry") or "").strip().lower()
                return bool(p_ind and j_ind and p_ind != j_ind)
        return False

    def _rule_company_size_mismatch(self, c: dict[str, Any]) -> bool:
        p_sz = (c.get("profile", {}).get("current_company_size") or "").strip()
        for j in c.get("career_history", []) or []:
            if j.get("is_current"):
                j_sz = (j.get("company_size") or "").strip()
                return bool(p_sz and j_sz and p_sz != j_sz)
        return False

    def _rule_title_vs_description(self, c: dict[str, Any]) -> bool:
        for j in c.get("career_history", []) or []:
            fam = title_family(j.get("title"))
            if fam is None:
                continue
            hits = _description_family_hits(j.get("description", ""))
            own_hits = hits.get(fam, 0)
            other_max = max((n for f, n in hits.items() if f != fam), default=0)
            if own_hits == 0 and other_max >= 3:
                return True
        return False

    def _rule_summary_vs_title(self, c: dict[str, Any]) -> bool:
        profile = c.get("profile") or {}
        summary = (profile.get("summary") or "").lower()
        current_fam = title_family(profile.get("current_title"))
        if not summary or not current_fam:
            return False
        for fam, patterns in _SUMMARY_PATTERNS.items():
            if fam == current_fam:
                continue
            for pattern in patterns:
                if pattern.search(summary):
                    return True
        return False

    def _rule_assessment_contradicts_proficiency(self, c: dict[str, Any]) -> bool:
        scores = (c.get("redrob_signals") or {}).get("skill_assessment_scores") or {}
        if not scores:
            return False
        for sk in c.get("skills") or []:
            name = sk.get("name")
            prof = (sk.get("proficiency") or "").lower()
            if prof in {"advanced", "expert"} and name in scores and scores[name] < 30:
                return True
        return False

    def _rule_unrelated_skill_cluster(self, c: dict[str, Any]) -> bool:
        names = {(s.get("name") or "").lower() for s in c.get("skills") or []}
        if not names:
            return False
        hit_buckets = sum(1 for toks in SKILL_BUCKETS.values() if names & toks)
        return hit_buckets >= SKILL_CLUSTER_THRESHOLD

    def _rule_too_many_skills_for_yoe(self, c: dict[str, Any]) -> bool:
        yoe = (c.get("profile") or {}).get("years_of_experience") or 0
        if yoe <= 0:
            return False
        return len(c.get("skills") or []) > 3 * yoe + 5

    # --- driver -----------------------------------------------------------

    # Static dispatch table: only methods listed here are ever called.
    # Disabled rules cost nothing.
    _RULE_FNS = {
        "industry_mismatch": "_rule_industry_mismatch",
        "company_size_mismatch": "_rule_company_size_mismatch",
        "title_vs_description": "_rule_title_vs_description",
        "summary_vs_title": "_rule_summary_vs_title",
        "assessment_contradicts_proficiency": "_rule_assessment_contradicts_proficiency",
        "unrelated_skill_cluster": "_rule_unrelated_skill_cluster",
        "too_many_skills_for_yoe": "_rule_too_many_skills_for_yoe",
    }

    def _active_rules(self) -> list[tuple[str, callable]]:
        return [
            (name, getattr(self, attr))
            for name, attr in self._RULE_FNS.items()
            if self.enabled.get(name, False)
        ]

    def _check(self, c: dict[str, Any], active: list[tuple[str, callable]]) -> list[str]:
        return [name for name, fn in active if fn(c)]

    def run(self, pool: list[CandidateContext]) -> list[CandidateContext]:
        active = self._active_rules()
        if not active:
            print("    cross_field_check: no rules enabled, skipping")
            return pool

        n_any = 0
        rule_counts: dict[str, int] = {name: 0 for name, _ in active}
        for ctx in pool:
            if ctx.is_dropped:
                continue
            hits = self._check(ctx.candidate, active)
            if not hits:
                continue
            n_any += 1
            penalty = 0.0
            for rule in hits:
                rule_counts[rule] += 1
                penalty += self.weights.get(rule, 0.0)
            ctx.components["honeypot_soft"] = (
                ctx.components.get("honeypot_soft", 0.0) + penalty
            )
            if penalty >= 0.3:
                ctx.notes.append(
                    f"Cross-field flags: {', '.join(hits)} (soft={penalty:.2f})"
                )

        active_names = {name for name, _ in active}
        print(
            f"    cross_field_check: {n_any:,} candidates have >=1 inconsistency "
            f"(rules active: {', '.join(active_names)})"
        )
        for rule, n in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
            print(f"        {rule:<40} {n:>6,}")
        return pool
