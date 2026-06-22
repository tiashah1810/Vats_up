"""
Shared heuristics used across multiple stages.

Kept tiny and dependency-free. If something here grows beyond a handful
of lookups, promote it to its own module.
"""

from __future__ import annotations

# Token bags per role family. Used both for classifying current_title
# (which family does this job belong to?) and for cross-checking
# description / summary text against the declared title.
#
# Designed to be conservative: each family includes only unambiguous
# domain tokens, so missed-classifications are common but mis-
# classifications are rare. Cross-field rules treat None / no-match as
# "skip this candidate" rather than "fail this candidate".
TITLE_TOKENS: dict[str, set[str]] = {
    "ai_ml": {
        "ml", "machine learning", "ai engineer", "ml engineer",
        "data scientist", "research scientist", "applied scientist",
        "nlp", "llm", "deep learning", "computer vision",
    },
    "data_eng": {
        "data engineer", "analytics engineer", "etl developer",
        "data platform", "data infrastructure", "pipeline",
    },
    "data_analyst": {"data analyst", "business analyst", "bi analyst", "analytics"},
    "backend": {
        "backend", "back-end", "back end", "server", "platform engineer",
        "infrastructure engineer", "api engineer",
    },
    "frontend": {"frontend", "front-end", "front end", "ui engineer", "web developer"},
    "fullstack": {"full stack", "full-stack", "fullstack"},
    "devops": {"devops", "sre", "site reliability", "cloud engineer", "platform"},
    "mobile": {"android", "ios", "mobile developer", "mobile engineer"},
    "qa": {"qa", "quality assurance", "test engineer", "sdet"},
    "security": {"security", "infosec", "appsec", "soc analyst"},
    "designer": {"designer", "ux", "ui designer", "product designer", "visual designer"},
    "pm": {"product manager", "program manager", "tpm"},
    "ops_mgmt": {"operations manager", "operations lead", "support manager"},
    "marketing": {"marketing", "brand", "growth", "seo specialist", "content writer"},
    "sales": {"sales", "account executive", "business development", "bdr", "sdr"},
    "hr": {"recruiter", "hr ", "talent acquisition", "people ops"},
    "finance": {"finance", "accountant", "controller", "auditor"},
    "mechanical": {"mechanical engineer", "design engineer", "cad engineer"},
    "electrical": {"electrical engineer", "electronics engineer", "embedded engineer"},
    "civil": {"civil engineer", "structural engineer"},
    "teacher": {"teacher", "lecturer", "professor", "instructor"},
    "health": {"nurse", "doctor", "physician", "pharmacist"},
}

# Per-family token bag for description / summary text matching. Looser
# than the title lookup above because free-text descriptions use
# different vocabulary than titles.
FAMILY_DESC_TOKENS: dict[str, set[str]] = {
    "ai_ml": {
        "model", "training", "embedding", "fine-tun", "neural", "transformer",
        "inference", "ml pipeline", "feature engineer",
    },
    "data_eng": {
        "pipeline", "airflow", "spark", "kafka", "warehouse", "etl", "dbt",
        "ingestion", "snowflake", "bigquery",
    },
    "data_analyst": {"dashboard", "metric", "kpi", "report", "analysis", "tableau", "powerbi"},
    "backend": {"api", "service", "endpoint", "database", "schema", "microservice"},
    "frontend": {"react", "vue", "angular", "component", "ui", "css"},
    "fullstack": {"frontend", "backend", "api", "react", "node"},
    "devops": {"kubernetes", "terraform", "ci/cd", "deploy", "infrastructure", "aws"},
    "mobile": {"android", "ios", "swift", "kotlin", "mobile app"},
    "qa": {"test", "automation", "selenium", "qa", "regression"},
    "security": {"vulnerab", "pentest", "soc", "siem", "audit"},
    "designer": {"design", "wireframe", "prototype", "figma", "sketch", "user research"},
    "pm": {"roadmap", "stakeholder", "requirement", "product launch", "user research"},
    "ops_mgmt": {"team of", "process", "escalation", "agents", "support"},
    "marketing": {"campaign", "brand", "audience", "content", "seo", "social"},
    "sales": {"quota", "pipeline", "deal", "client", "revenue", "prospect"},
    "hr": {"candidate", "interview", "hiring", "talent"},
    "finance": {"ledger", "audit", "tax", "balance sheet", "financial"},
    "mechanical": {"mechanical", "cad", "solidworks", "manufactur", "hardware", "fea", "ansys"},
    "electrical": {"circuit", "pcb", "firmware", "embedded", "voltage", "signal"},
    "civil": {"structural", "construction", "concrete", "load"},
    "teacher": {"student", "curriculum", "lesson", "classroom", "exam"},
    "health": {"patient", "clinical", "ward", "hospital", "treatment"},
}


def title_family(title: str | None) -> str | None:
    """Return the role family for a free-text title, or None if ambiguous.

    Resolution rules:
      * lowercase substring match against TITLE_TOKENS
      * if multiple families match, prefer the longest matching token
        (so "Senior Backend Engineer" -> backend, not the generic "engineer")
      * return None if nothing matches; callers should treat None as
        "skip" rather than as a failure mode.
    """
    if not title:
        return None
    t = title.lower()
    best_family: str | None = None
    best_len = 0
    for family, tokens in TITLE_TOKENS.items():
        for tok in tokens:
            if tok in t and len(tok) > best_len:
                best_family = family
                best_len = len(tok)
    return best_family
