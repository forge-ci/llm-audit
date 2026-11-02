"""Image results are billed by pixels, not by the base64 payload.

A screenshot Read is the one place where transcript bytes and context tokens
diverge by two orders of magnitude. Counting the payload made `file` look like
the fleet's dominant context cost on 2026-11-01 -- 1.27 GB of a 2.2 GB total --
when the images were under 2 % of the tokens, so the audit pointed every
optimisation at the wrong subsystem.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

from llm_audit.aggregate import render_report
from llm_audit.transcript import (
    BYTES_PER_TOKEN,
    DEFAULT_IMAGE_TOKENS,
    SessionReport,
    analyze_session,
    image_tokens,
    read_session,
    result_bytes,
    result_detail,
)


def png(width: int, height: int, filler: int = 0) -> str:
    """A header-only PNG: enough for the IHDR the audit parses."""
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    header += struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return base64.b64encode(header + b"\x00" * filler).decode()


def jpeg(width: int, height: int) -> str:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", height, width)
    sof0 += b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return base64.b64encode(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9").decode()


def image_result(data: str, media: str = "image/png") -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": "t0",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
        ],
    }


def test_png_tokens_come_from_pixels() -> None:
    # 390x1600 phone screenshot: 832 tokens -- the real screenshots in the
    # 2026-11-01 window were exactly this shape.
    assert image_tokens({"data": png(390, 1600)}) == 832


def test_jpeg_tokens_skip_the_app_segment() -> None:
    assert image_tokens({"data": jpeg(1440, 1000)}) == 1920


def test_unparseable_image_falls_back_to_a_screenshot() -> None:
    assert image_tokens({"data": "not-base64!!"}) == DEFAULT_IMAGE_TOKENS
    assert image_tokens({}) == DEFAULT_IMAGE_TOKENS


def test_image_payload_is_not_text_bytes() -> None:
    payload = png(390, 1600, filler=240_000)
    context_bytes, tokens, raw = result_detail(image_result(payload))
    assert tokens == 832
    assert raw > 300_000  # what the old accounting charged
    assert context_bytes == 832 * BYTES_PER_TOKEN
    assert context_bytes < raw / 50


def test_text_results_are_unchanged() -> None:
    block = {"type": "tool_result", "tool_use_id": "t0", "content": "x" * 4_000}
    assert result_detail(block) == (4_000, 0, 0)
    assert result_bytes(block) == 4_000


def test_session_counts_images_separately(tmp_path: Path) -> None:
    transcript = tmp_path / "s.jsonl"
    payload = png(390, 1600, filler=240_000)
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "cwd": "/repo",
                "message": {
                    "id": "m0",
                    "role": "assistant",
                    "usage": {"input_tokens": 5},
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t0",
                            "name": "Read",
                            "input": {"file_path": "/x.png"},
                        }
                    ],
                },
            }
        ),
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": [image_result(payload)]}}
        ),
    ]
    transcript.write_text("\n".join(lines) + "\n")

    read = read_session(transcript)
    report = analyze_session(read, transcript, "t", "main")
    assert report.image_results == 1
    assert report.image_tokens == 832
    assert report.image_payload_bytes > 300_000
    # The category total follows the vision cost, not the payload.
    assert report.bytes_by_cat["file:read"] == 832 * BYTES_PER_TOKEN


def test_report_states_the_correction(tmp_path: Path) -> None:
    rows = [
        SessionReport(
            file="/t/s.jsonl",
            session_id="s",
            corpus="t",
            kind="main",
            cwd="/repo",
            responses=2,
            compactions=0,
            tool_calls=1,
            tool_results=1,
            result_bytes=3_328,
            exposure_bytes=6_656,
            image_results=1,
            image_tokens=832,
            image_payload_bytes=309_000,
            sidechain_results=0,
            sidechain_bytes=0,
            usage=_usage(),
            calls_by_cat={"file:read": 1},
            bytes_by_cat={"file:read": 3_328},
            exposure_by_cat={"file:read": 6_656},
            calls_by_tool={"Read": 1},
            bytes_by_tool={"Read": 3_328},
            exposure_by_tool={"Read": 6_656},
            order_runs=[],
            share_calls={},
            share_bytes={},
            share_exposure={},
            dup_calls=0,
            dup_call_count=0,
            dup_waste_bytes=0,
            dup_waste_exposure=0,
            replayed_outputs=0,
            replayed_out_bytes=0,
            norm_repeat_calls=0,
            norm_repeat_bytes=0,
            norm_repeat_exposure=0,
            norm_repeat_by_cat={},
            empty_calls=0,
            empty_bytes=0,
            empty_exposure=0,
            empty_by_cat={},
            avoidable_requests=0,
            avoidable_tokens=0,
            top_dups=[],
            top_biggest=[],
            top_replayed=[],
            norm_repeat_examples=[],
            signatures={},
        )
    ]
    text = render_report(rows)
    assert "image results          : 1 at vision cost" in text
    assert "1k tokens" in text
    assert "309.0KB of base64 payload is not text" in text


def _usage() -> Any:
    from llm_audit.transcript import Usage

    return Usage(input=10, cache_read=100)
