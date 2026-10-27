"""Time windows: parsing, and what a windowed read actually counts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_audit.corpus import discover
from llm_audit.transcript import analyze_session, read_session
from llm_audit.window import describe, in_window, parse_since, parse_timestamp

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)


def test_relative_specs() -> None:
    assert parse_since("24h", NOW) == NOW - timedelta(hours=24)
    assert parse_since("90m", NOW) == NOW - timedelta(minutes=90)
    assert parse_since("7d", NOW) == NOW - timedelta(days=7)
    assert parse_since("2w", NOW) == NOW - timedelta(weeks=2)


def test_bare_number_is_hours() -> None:
    assert parse_since("48", NOW) == NOW - timedelta(hours=48)


def test_iso_forms() -> None:
    assert parse_since("2026-10-26", NOW) == datetime(2026, 10, 26, tzinfo=UTC)
    assert parse_since("2026-10-26T09:30:00Z", NOW) == datetime(2026, 10, 26, 9, 30, tzinfo=UTC)


def test_no_window() -> None:
    assert parse_since(None) is None
    assert parse_since("all") is None
    assert parse_since("") is None


def test_invalid_spec_is_loud() -> None:
    with pytest.raises(ValueError, match="unrecognised --since"):
        parse_since("yesterday-ish")


def test_parse_timestamp_forms() -> None:
    assert parse_timestamp("2026-10-27T06:40:10.481Z") == datetime(
        2026, 10, 27, 6, 40, 10, 481000, tzinfo=UTC
    )
    assert parse_timestamp("2026-10-27T06:40:10+00:00") == datetime(
        2026, 10, 27, 6, 40, 10, tzinfo=UTC
    )
    # a naive stamp is assumed UTC rather than dropped
    assert parse_timestamp("2026-10-27T06:40:10") == datetime(2026, 10, 27, 6, 40, 10, tzinfo=UTC)
    assert parse_timestamp(None) is None
    assert parse_timestamp("not a time") is None


def test_in_window_semantics() -> None:
    cutoff = NOW - timedelta(hours=24)
    assert in_window("2026-10-27T06:00:00Z", None)  # no window admits everything
    assert in_window("2026-10-27T06:00:00Z", cutoff)
    assert not in_window("2026-10-20T06:00:00Z", cutoff)
    # an undatable record must never be assumed to be inside the window
    assert not in_window(None, cutoff)


def test_describe() -> None:
    assert describe(None) == "all time"
    assert describe(NOW).startswith("since 2026-10-27T12:00:00Z")


def write_timed(path: Path, rows: list[tuple[str, str, int]]) -> None:
    """rows: (iso_timestamp, command, result_size)."""
    lines: list[str] = []
    for index, (ts, command, size) in enumerate(rows):
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "cwd": "/x",
                    "timestamp": ts,
                    "message": {
                        "id": f"m{index}",
                        "role": "assistant",
                        "model": "m",
                        "usage": {"input_tokens": 10},
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"t{index}",
                                "name": "Bash",
                                "input": {"command": command},
                            }
                        ],
                    },
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "timestamp": ts,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t{index}",
                                "content": "x" * size,
                            }
                        ],
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def test_window_counts_only_records_inside_it(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_timed(
        target,
        [
            ("2026-10-20T00:00:00Z", "cargo build", 1000),  # 7 days ago
            ("2026-10-20T00:10:00Z", "cargo build", 1000),  # 7 days ago
            ("2026-10-27T00:00:00Z", "gh run view 1", 500),  # inside 24h
            ("2026-10-27T00:05:00Z", "gh run view 1", 500),  # inside 24h
        ],
    )
    cutoff = datetime(2026, 10, 26, 12, 0, 0, tzinfo=UTC)

    full = read_session(target, "main")
    windowed = read_session(target, "main", cutoff)

    assert full.responses == 4
    assert windowed.responses == 2

    full_report = analyze_session(full, target, "t", "main")
    report = analyze_session(windowed, target, "t", "main")
    assert full_report.result_bytes == 3000
    assert report.result_bytes == 1000
    assert set(report.calls_by_cat) == {"gh:run"}
    # exposure is computed inside the window: 500 bytes x 1 remaining request
    assert report.exposure_bytes == 500
    assert report.responses == 2


def test_discover_skips_files_older_than_the_window(tmp_path: Path) -> None:
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    for path in (old, new):
        write_timed(path, [("2026-10-27T00:00:00Z", "cargo test", 10)])
    stale = (NOW - timedelta(days=3)).timestamp()
    os.utime(old, (stale, stale))

    cutoff = NOW - timedelta(hours=24)
    assert [j.path for j in discover([tmp_path], since=cutoff)] == [new]
    assert len(discover([tmp_path], since=None)) == 2
