"""Exposure and waste-ledger arithmetic.

Every number here is checkable by hand, which is the point: the exposure
multiplier and the avoidable-request index are both easy to get off by one.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_audit.transcript import SessionReport, analyze_session, read_session


def write_transcript(
    path: Path,
    responses: list[dict[str, object]],
    result_sizes: list[int],
    *,
    sidechain: bool = False,
) -> None:
    """Build a minimal transcript: N assistant rows, N-1..N result rows."""
    lines: list[str] = []
    for index, usage in enumerate(responses):
        record: dict[str, object] = {
            "type": "assistant",
            "cwd": "/x",
            "message": {
                "id": f"m{index}",
                "role": "assistant",
                "model": "test-model",
                "usage": usage,
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"t{index}",
                        "name": "Bash",
                        "input": {"command": "gh pr view 1"},
                    }
                ],
            },
        }
        if sidechain:
            record["isSidechain"] = True
        lines.append(json.dumps(record))
        if index < len(result_sizes):
            result: dict[str, object] = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{index}",
                            "content": "x" * result_sizes[index],
                        }
                    ],
                },
            }
            if sidechain:
                result["isSidechain"] = True
            lines.append(json.dumps(result))
    path.write_text("\n".join(lines) + "\n")


def analyze(path: Path, kind: str = "main") -> SessionReport:
    return analyze_session(read_session(str(path), kind), str(path), "test", kind)


def test_exposure_multiplies_by_later_requests(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_transcript(target, [{"input_tokens": 1, "output_tokens": 1}] * 3, [100, 100])
    row = analyze(target)
    assert row.responses == 3
    assert row.result_bytes == 200
    # the first result is re-sent by requests 2 and 3 -> 2x
    # the second only by request 3 -> 1x, so 100*2 + 100*1
    assert row.exposure_bytes == 300


def test_duplicate_calls_are_counted(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_transcript(target, [{"input_tokens": 1}] * 4, [50, 50, 50, 50])
    row = analyze(target)
    counts = row.signatures["Bash::gh pr view 1"]
    assert counts["calls"] == 4
    assert row.dup_waste_exposure > 0


def test_subagent_transcript_is_counted(tmp_path: Path) -> None:
    """A subagent's own file is sidechain top to bottom, and still billed."""
    target = tmp_path / "s.jsonl"
    write_transcript(
        target,
        [{"input_tokens": 1}, {"input_tokens": 1}],
        [80, 80],
        sidechain=True,
    )
    assert analyze(target, "main").result_bytes == 0
    subagent = analyze(target, "subagent")
    assert subagent.result_bytes == 160
    assert subagent.calls_by_cat == {"gh:pr": 2}


def test_empty_results_are_avoidable_requests(tmp_path: Path) -> None:
    """`sleep 60` returns nothing: it must be counted as zero information."""
    target = tmp_path / "s.jsonl"
    write_transcript(target, [{"input_tokens": 1}] * 4, [0, 0, 0, 0])
    row = analyze(target)
    assert row.empty_calls == 4
    assert row.avoidable_requests == 4
    assert row.avoidable_tokens == 4


def test_informative_request_is_not_avoidable(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_transcript(target, [{"input_tokens": 1}] * 2, [10, 20])
    assert analyze(target).avoidable_requests == 0


def test_round_trip_through_json(tmp_path: Path) -> None:
    target = tmp_path / "s.jsonl"
    write_transcript(target, [{"input_tokens": 5}] * 2, [10, 20])
    row = analyze(target)
    restored = SessionReport.from_json(row.to_json())
    assert restored.exposure_bytes == row.exposure_bytes
    assert restored.avoidable_tokens == row.avoidable_tokens
    assert restored.signatures.keys() == row.signatures.keys()
