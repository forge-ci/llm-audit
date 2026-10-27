"""Re-read, poll and position analyses."""

from __future__ import annotations

from pathlib import Path

from helpers import bash, edit, write_calls
from llm_audit.analysis import analyze_polls, analyze_rereads, analyze_when
from llm_audit.corpus import Job


def jobs(*paths: Path) -> list[Job]:
    return [Job(str(p), "test", "main") for p in paths]


def test_reread_without_an_edit_is_gratuitous(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_calls(
        target,
        [
            ("Bash", bash("sed -n 1,80p crates/x/src/lib.rs"), "one"),
            ("Bash", bash("sed -n 90,160p crates/x/src/lib.rs"), "two"),
        ],
    )
    verdict = analyze_rereads(jobs(target))
    assert verdict.reads == 2
    assert verdict.first_reads == 1
    assert verdict.gratuitous == 1
    assert verdict.justified == 0


def test_reread_after_an_edit_is_justified(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_calls(
        target,
        [
            ("Bash", bash("sed -n 1,80p crates/x/src/lib.rs"), "one"),
            ("Edit", edit("/repo/crates/x/src/lib.rs"), "ok"),
            ("Bash", bash("sed -n 1,80p crates/x/src/lib.rs"), "two"),
        ],
    )
    verdict = analyze_rereads(jobs(target))
    assert verdict.justified == 1
    assert verdict.gratuitous == 0


def test_bare_filename_is_not_counted(tmp_path: Path) -> None:
    """`lib.rs` alone cannot be disambiguated, so it must be excluded."""
    target = tmp_path / "s.jsonl"
    write_calls(
        target,
        [
            ("Bash", bash("sed -n 1,80p lib.rs"), "one"),
            ("Bash", bash("sed -n 90,160p lib.rs"), "two"),
        ],
    )
    assert analyze_rereads(jobs(target)).reads == 0


def test_pending_polls_are_separated_from_terminal(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_calls(
        target,
        [
            ("Bash", bash("gh run view 1 --json status,conclusion"), "queued"),
            ("Bash", bash("gh run view 1 --json status,conclusion"), "in_progress"),
            ("Bash", bash("gh run view 1 --json status,conclusion"), "queued"),
            ("Bash", bash("gh run view 1 --json status,conclusion"), "completed success"),
        ],
    )
    verdict = analyze_polls(jobs(target))
    assert verdict.calls == 4
    assert verdict.pending == 3
    assert verdict.terminal == 1
    assert verdict.streaks[0][0] == 3


def test_non_poll_commands_are_ignored(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_calls(target, [("Bash", bash("cargo test"), "ok")])
    assert analyze_polls(jobs(target)).calls == 0


def test_exposure_concentrates_early(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    # one big early result, then many cheap ones to give it a long tail
    calls: list[tuple[str, dict[str, object], str]] = [
        ("Bash", bash("cat crates/x/src/big.rs"), "B" * 4000)
    ]
    for index in range(30):
        calls.append(("Bash", bash(f"echo {index}"), "s"))
    write_calls(target, calls)
    verdict = analyze_when(jobs(target))
    assert verdict.sessions == 1
    assert verdict.buckets[0] > verdict.buckets[3]
