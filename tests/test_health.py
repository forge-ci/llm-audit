"""The gate: thresholds, and the output must be deterministic.

A cron ``monitor`` diffs this output byte for byte. If it contains timestamps,
counts that drift as the corpus grows, or anything nondeterministic, the gate
trips on every tick and the agent it guards runs every day.
"""

from __future__ import annotations

from pathlib import Path

from helpers import bash, write_calls
from llm_audit.aggregate import totals
from llm_audit.health import evaluate, render_verdict
from llm_audit.transcript import SessionReport, analyze_session, read_session


def build_rows(tmp_path: Path, name: str, command: str, text: str) -> SessionReport:
    target = tmp_path / name
    write_calls(target, [("Bash", bash(command), text)] * 3)
    return analyze_session(read_session(str(target), "main"), str(target), "corpus-a", "main")


def test_verdict_has_no_decimals_or_clock(tmp_path: Path) -> None:
    row = build_rows(tmp_path, "s.jsonl", "cargo test", "ok")
    gate = evaluate([row], [row], check_versions=False)
    verdict = render_verdict(gate)
    assert "verdict:" in verdict
    for line in verdict.splitlines():
        if line.startswith(("avoidable_pct:", "top_category:", "reread_pct:")):
            assert "." not in line, f"unrounded value would drift: {line}"
    assert "20" not in verdict.splitlines()[0]


def test_verdict_is_deterministic(tmp_path: Path) -> None:
    row = build_rows(tmp_path, "s.jsonl", "gh run view 1", "queued")
    first = render_verdict(evaluate([row], [row], check_versions=False))
    second = render_verdict(evaluate([row], [row], check_versions=False))
    assert first == second


def test_top_category_rule_fires(tmp_path: Path) -> None:
    row = build_rows(tmp_path, "s.jsonl", "gh run view 1", "queued")
    gate = evaluate([row], [row], check_versions=False)
    # one subsystem owns everything here, which is over the 30% threshold
    assert gate.top_category == "gh"
    assert any(f.rule == "top_category_pct" for f in gate.findings)


def test_avoidable_rise_fires_against_a_baseline(tmp_path: Path) -> None:
    row = build_rows(tmp_path, "s.jsonl", "sleep 60", "")
    gate = evaluate([row], [row], {"avoidable_pct": 50.0}, check_versions=False)
    assert any(f.rule == "avoidable_pct_delta" for f in gate.findings)


def test_delta_rule_needs_a_baseline(tmp_path: Path) -> None:
    """A first run has nothing to rise from, so the delta rule must stay quiet."""
    row = build_rows(tmp_path, "s.jsonl", "sleep 60", "")
    first = evaluate([row], [row], {}, check_versions=False)
    assert not any(f.rule == "avoidable_pct_delta" for f in first.findings)

    # with a real baseline, the same rows may legitimately fire
    later = evaluate([row], [row], {"avoidable_pct": 0.0}, check_versions=False)
    assert any(f.rule == "avoidable_pct_delta" for f in later.findings)


def test_new_signature_needs_a_baseline(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_calls(target, [("Bash", bash("cargo test"), "ok")] * 25)
    row = analyze_session(read_session(str(target), "main"), str(target), "corpus-a", "main")
    # first run: no baseline, so no new_signature finding even at 25 calls
    assert not any(
        f.rule == "new_signature_calls"
        for f in evaluate([row], [row], {}, check_versions=False).findings
    )
    # second run with a baseline that cannot contain the signature
    after = evaluate([row], [row], {"known_signatures": []}, check_versions=False)
    assert any(f.rule == "new_signature_calls" for f in after.findings)


def test_totals_are_honest_on_an_empty_report(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    target.write_text("")
    row = analyze_session(read_session(str(target), "main"), str(target), "c", "main")
    assert totals([row]).exposure == 0
    assert totals([row]).requests == 0
