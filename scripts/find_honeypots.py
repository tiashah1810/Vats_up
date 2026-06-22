#!/usr/bin/env python3
"""
Honeypot detector for the Redrob Intelligent Candidate Discovery & Ranking
Challenge.

The hackathon spec (submission_spec section 7) states the candidate pool
contains ~80 honeypots with "subtly impossible profiles". Submissions with
honeypot rate > 10% in the top 100 are disqualified.

Design principles
-----------------
* HONEYPOTS ARE INTERNAL CONTRADICTIONS, not weak fits. We only flag
  arithmetic / chronological / categorical impossibilities that no real
  candidate could honestly satisfy.
* PRECISION OVER RECALL. With ~80 honeypots in 100k, every false positive
  could burn a strong real candidate from our top 100. So every rule
  below was validated against the dataset to keep base-rate noise near
  zero. Rules that fire on >1% of the pool are treated as generator
  noise (e.g. salary min>max fires on 18.9% of candidates -> dropped).
* HARD vs SOFT tiers. HARD rules are the canonical impossibilities the
  spec itself calls out; even a single HARD hit marks a honeypot. SOFT
  rules are corroborating signals that are too noisy to use alone.

Strong rules (HARD severity)
----------------------------
* skill.proficient_zero_duration .... advanced/expert skill with duration=0
                                      (matches the spec's canonical example)
* skill.expert_too_short ............ expert proficiency with <6 months
* skill.duration_far_exceeds_career . skill duration >> total years worked
* career.yoe_exceeds_span ........... claimed YOE > career calendar span
                                      ("8 yrs at company founded 3 yrs ago")
* career.worked_far_exceeds_yoe ..... worked time >> claimed YOE
* career.start_after_end ............ a job has start_date > end_date
* career.multiple_current_overlap ... 2+ is_current jobs with overlapping
                                      calendars
* career.is_current_with_end_date ... is_current=True yet end_date is set
* career.not_current_no_end_date .... is_current=False yet end_date null
* profile.current_company_mismatch .. profile.current_company != the
                                      is_current=True job's company
* profile.current_title_mismatch .... same, for title
* education.start_after_end ......... start_year > end_year
* education.end_year_too_far_future . end_year > anchor_year + 5
* signals.last_active_in_future ..... last_active beyond dataset anchor
* signals.signup_in_future .......... signup beyond dataset anchor
* signals.notice_period_out_of_range  outside the schema-declared 0-180
* signals.interview_completion_rate_out_of_range ... outside 0-1
* signals.offer_acceptance_rate_out_of_range ....... outside [-1, 1]
* signals.recruiter_response_rate_out_of_range ..... outside 0-1
* signals.applications_absurd ....... > 1000 applications in 30 days

Rules deliberately NOT used (validated as generator noise)
----------------------------------------------------------
* signals.salary_min_gt_max .......... fires on 18.9% of pool
* signals.saved_gt_views ............. fires on 7.7%
* signals.signup_after_last_active ... fires on 7.5%
* career.duration_mismatch (current jobs) ... fires on 100% (anchor mismatch)

Usage
-----
    uv run python scripts/find_honeypots.py \
        --candidates data/India_runs_data_and_ai_challenge/candidates.jsonl \
        --out artifacts/honeypots.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

# --- Constants -------------------------------------------------------------

# Dataset is anchored around mid-2026 (sample candidates show last_active
# dates up to 2026-05-20 and current-job durations consistent with a "now"
# of early-to-mid 2026). Use a comfortable upper bound for future-date
# checks; this affects only HARD impossibility rules.
DATASET_ANCHOR = date(2026, 12, 31)

# Months by which claimed YOE may exceed the calendar span of a person's
# career before we treat it as impossible (~1.5y absorbs rounding and gaps).
YOE_OVER_SPAN_TOLERANCE_MONTHS = 18

# Months by which actually-worked time may exceed claimed YOE before we
# flag. Large because the generator is sloppy here; only extreme cases
# (4+ years over) are honeypots.
WORKED_OVER_YOE_HARD_THRESHOLD = 48

# Months a skill's duration may exceed the person's total career length
# before being treated as impossible. Large slack (5y) because skills
# can predate paid work via school / hobby projects.
SKILL_OVER_CAREER_HARD_THRESHOLD_MONTHS = 60

# Tolerance for an ended job's duration_months vs (end - start).
ENDED_JOB_DURATION_TOLERANCE_MONTHS = 3

# Future-education slack (PhDs / part-time programs).
FUTURE_EDUCATION_SLACK_YEARS = 5

# Calendar-day overlap below which multiple "is_current" jobs are ignored.
OVERLAP_TOLERANCE_DAYS = 31

HARD = "HARD"
SOFT = "SOFT"

# rule_id -> severity. Anything not listed defaults to SOFT.
RULE_SEVERITY: dict[str, str] = {
    "skill.proficient_zero_duration": HARD,
    "skill.expert_too_short": HARD,
    "skill.duration_far_exceeds_career": HARD,
    "career.yoe_exceeds_span": HARD,
    "career.worked_far_exceeds_yoe": HARD,
    "career.start_after_end": HARD,
    "career.multiple_current_overlap": HARD,
    "career.is_current_with_end_date": HARD,
    "career.not_current_no_end_date": HARD,
    "career.ended_job_duration_mismatch": SOFT,
    "profile.current_company_mismatch": HARD,
    "profile.current_title_mismatch": HARD,
    "education.start_after_end": HARD,
    "education.end_year_too_far_future": HARD,
    "signals.last_active_in_future": HARD,
    "signals.signup_in_future": HARD,
    "signals.notice_period_out_of_range": HARD,
    "signals.interview_completion_rate_out_of_range": HARD,
    "signals.offer_acceptance_rate_out_of_range": HARD,
    "signals.recruiter_response_rate_out_of_range": HARD,
    "signals.applications_absurd": HARD,
}


# --- Helpers --------------------------------------------------------------


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _months_between(start: date, end: date) -> int:
    return round((end - start).days / 30.4375)


def _overlap_days(a_start: date, a_end: date, b_start: date, b_end: date) -> int:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0, (hi - lo).days)


# --- Rule families --------------------------------------------------------


def check_career(c: dict[str, Any], violations: list[tuple[str, str]]) -> None:
    history = c.get("career_history") or []
    profile = c.get("profile") or {}
    yoe = profile.get("years_of_experience")

    current_jobs: list[dict[str, Any]] = []
    intervals: list[tuple[date, date]] = []

    for idx, job in enumerate(history):
        start = _parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = _parse_date(end_raw)
        is_current = bool(job.get("is_current"))
        dm = job.get("duration_months")

        if is_current and end_raw not in (None, ""):
            violations.append((
                "career.is_current_with_end_date",
                f"job[{idx}] is_current=True but end_date={end_raw!r}",
            ))
        if not is_current and end_raw in (None, ""):
            violations.append((
                "career.not_current_no_end_date",
                f"job[{idx}] is_current=False but end_date is null",
            ))

        if start and end and start > end:
            violations.append((
                "career.start_after_end",
                f"job[{idx}] start={start} > end={end}",
            ))

        # Duration check only for ENDED jobs (both dates present). For
        # current jobs the generator uses a different anchor than
        # wall-clock today, which makes the check unreliable.
        if start and end and isinstance(dm, (int, float)):
            actual = _months_between(start, end)
            if abs(actual - dm) > ENDED_JOB_DURATION_TOLERANCE_MONTHS:
                violations.append((
                    "career.ended_job_duration_mismatch",
                    f"job[{idx}] duration_months={dm} but dates imply ~{actual}",
                ))

        if is_current:
            current_jobs.append(job)

        eff_end = end if end else (DATASET_ANCHOR if is_current else None)
        if start and eff_end:
            intervals.append((start, eff_end))

    if len(current_jobs) > 1:
        bad_overlap = False
        for i in range(len(current_jobs)):
            for j in range(i + 1, len(current_jobs)):
                s1 = _parse_date(current_jobs[i].get("start_date"))
                s2 = _parse_date(current_jobs[j].get("start_date"))
                if s1 and s2 and _overlap_days(s1, DATASET_ANCHOR, s2, DATASET_ANCHOR) > OVERLAP_TOLERANCE_DAYS:
                    bad_overlap = True
        if bad_overlap:
            violations.append((
                "career.multiple_current_overlap",
                f"{len(current_jobs)} is_current jobs with overlapping calendars",
            ))

    if current_jobs:
        cj = current_jobs[0]
        pc = (profile.get("current_company") or "").strip().lower()
        pt = (profile.get("current_title") or "").strip().lower()
        jc = (cj.get("company") or "").strip().lower()
        jt = (cj.get("title") or "").strip().lower()
        if pc and jc and pc != jc:
            violations.append((
                "profile.current_company_mismatch",
                f"profile={pc!r} vs current job={jc!r}",
            ))
        if pt and jt and pt != jt:
            violations.append((
                "profile.current_title_mismatch",
                f"profile={pt!r} vs current job={jt!r}",
            ))

    if intervals and isinstance(yoe, (int, float)) and yoe > 0:
        earliest = min(s for s, _ in intervals)
        latest = max(e for _, e in intervals)
        span_months = _months_between(earliest, latest)
        claimed_months = yoe * 12
        if claimed_months - span_months > YOE_OVER_SPAN_TOLERANCE_MONTHS:
            violations.append((
                "career.yoe_exceeds_span",
                f"YOE={yoe} (~{int(claimed_months)}m) > career span ~{span_months}m",
            ))

        merged: list[tuple[date, date]] = []
        for s, e in sorted(intervals):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        worked_months = sum(_months_between(s, e) for s, e in merged)
        if worked_months - claimed_months > WORKED_OVER_YOE_HARD_THRESHOLD:
            violations.append((
                "career.worked_far_exceeds_yoe",
                f"worked ~{worked_months}m but YOE only {yoe}",
            ))


def check_skills(c: dict[str, Any], violations: list[tuple[str, str]]) -> None:
    skills = c.get("skills") or []
    profile = c.get("profile") or {}
    yoe = profile.get("years_of_experience") or 0
    yoe_months = int(yoe * 12)

    for idx, sk in enumerate(skills):
        prof = (sk.get("proficiency") or "").lower()
        dm = sk.get("duration_months")
        name = sk.get("name", f"#{idx}")
        if not isinstance(dm, (int, float)):
            continue

        if prof in {"advanced", "expert"} and dm == 0:
            violations.append((
                "skill.proficient_zero_duration",
                f"skill={name!r} prof={prof} dur=0",
            ))

        if prof == "expert" and 0 < dm < 6:
            violations.append((
                "skill.expert_too_short",
                f"skill={name!r} expert with only {dm}m",
            ))

        if yoe_months and dm - yoe_months > SKILL_OVER_CAREER_HARD_THRESHOLD_MONTHS:
            violations.append((
                "skill.duration_far_exceeds_career",
                f"skill={name!r} duration={dm}m vs career {yoe_months}m",
            ))


def check_education(c: dict[str, Any], violations: list[tuple[str, str]]) -> None:
    cap = DATASET_ANCHOR.year + FUTURE_EDUCATION_SLACK_YEARS
    for idx, ed in enumerate(c.get("education") or []):
        sy = ed.get("start_year")
        ey = ed.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and sy > ey:
            violations.append((
                "education.start_after_end",
                f"education[{idx}] start={sy} > end={ey}",
            ))
        if isinstance(ey, int) and ey > cap:
            violations.append((
                "education.end_year_too_far_future",
                f"education[{idx}] end_year={ey}",
            ))


def check_signals(c: dict[str, Any], violations: list[tuple[str, str]]) -> None:
    sig = c.get("redrob_signals") or {}

    last = _parse_date(sig.get("last_active_date"))
    signup = _parse_date(sig.get("signup_date"))
    if last and last > DATASET_ANCHOR:
        violations.append((
            "signals.last_active_in_future",
            f"last_active={last} > anchor={DATASET_ANCHOR}",
        ))
    if signup and signup > DATASET_ANCHOR:
        violations.append((
            "signals.signup_in_future",
            f"signup={signup} > anchor={DATASET_ANCHOR}",
        ))

    np_days = sig.get("notice_period_days")
    if isinstance(np_days, (int, float)) and not (0 <= np_days <= 180):
        violations.append(("signals.notice_period_out_of_range", f"value={np_days}"))

    icr = sig.get("interview_completion_rate")
    if isinstance(icr, (int, float)) and not (0 <= icr <= 1):
        violations.append(("signals.interview_completion_rate_out_of_range", f"value={icr}"))

    oar = sig.get("offer_acceptance_rate")
    if isinstance(oar, (int, float)) and not (-1 <= oar <= 1):
        violations.append(("signals.offer_acceptance_rate_out_of_range", f"value={oar}"))

    rrr = sig.get("recruiter_response_rate")
    if isinstance(rrr, (int, float)) and not (0 <= rrr <= 1):
        violations.append(("signals.recruiter_response_rate_out_of_range", f"value={rrr}"))

    apps = sig.get("applications_submitted_30d")
    if isinstance(apps, int) and apps > 1000:
        violations.append(("signals.applications_absurd", f"value={apps}"))


# --- Driver ---------------------------------------------------------------


def audit_candidate(c: dict[str, Any]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    check_career(c, violations)
    check_skills(c, violations)
    check_education(c, violations)
    check_signals(c, violations)
    return violations


def classify_severity(violations: list[tuple[str, str]]) -> str:
    if any(RULE_SEVERITY.get(r, SOFT) == HARD for r, _ in violations):
        return HARD
    return SOFT


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--candidates",
        default="data/India_runs_data_and_ai_challenge/candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz).",
    )
    p.add_argument(
        "--out",
        default="artifacts/honeypots.csv",
        help="Output CSV of flagged candidates.",
    )
    p.add_argument(
        "--hard-only",
        action="store_true",
        help="Only write candidates with at least one HARD-severity violation.",
    )
    args = p.parse_args()

    src = Path(args.candidates)
    if not src.exists():
        print(f"error: candidates file not found: {src}", file=sys.stderr)
        return 1

    if src.suffix == ".gz":
        import gzip

        opener = lambda: gzip.open(src, "rt", encoding="utf-8")
    else:
        opener = lambda: open(src, "r", encoding="utf-8")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rule_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    total = 0

    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                cand = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warn: bad JSON on line {total}: {e}", file=sys.stderr)
                continue

            v = audit_candidate(cand)
            if not v:
                continue
            sev = classify_severity(v)
            if args.hard_only and sev != HARD:
                continue
            for rule, _ in v:
                rule_counts[rule] += 1
            rows.append({
                "candidate_id": cand.get("candidate_id", ""),
                "severity": sev,
                "num_violations": len(v),
                "num_hard": sum(1 for r, _ in v if RULE_SEVERITY.get(r, SOFT) == HARD),
                "rules": ";".join(sorted({r for r, _ in v})),
                "details": " | ".join(f"{r}: {d}" for r, d in v),
            })

    # HARD first, then by hard-count desc, then total violations, then id.
    rows.sort(
        key=lambda r: (
            0 if r["severity"] == HARD else 1,
            -r["num_hard"],
            -r["num_violations"],
            r["candidate_id"],
        )
    )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "severity",
                "num_violations",
                "num_hard",
                "rules",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    hard_count = sum(1 for r in rows if r["severity"] == HARD)
    soft_count = sum(1 for r in rows if r["severity"] == SOFT)

    print(f"Scanned {total:,} candidates")
    print(f"Flagged {len(rows):,}  (HARD={hard_count:,}, SOFT={soft_count:,})  -> {out_path}")
    print(f"HARD/total = {hard_count / max(total, 1):.4%}   (~80 honeypots expected in 100k)")
    print("\nViolations by rule:")
    for rule, n in rule_counts.most_common():
        tag = "HARD" if RULE_SEVERITY.get(rule, SOFT) == HARD else "soft"
        print(f"  {n:6d}  [{tag}]  {rule}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
