"""Aggregate many session reports into the ranked report.

Ranking is by exposure, split by ``owner:subcommand`` so ``gh:issue`` and
``gh:run`` are separate rows rather than one pooled "gh".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .classify import coarse
from .transcript import SessionReport, Usage


def merge(rows: Iterable[SessionReport], field: str) -> Counter[str]:
    """Sum one ``*_by_cat`` / ``*_by_tool`` mapping across sessions."""
    out: Counter[str] = Counter()
    for row in rows:
        values = getattr(row, field, None)
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, int):
                    out[str(key)] += value
    return out


def rollup(counter: Counter[str]) -> Counter[str]:
    """Roll fine categories up to owners."""
    out: Counter[str] = Counter()
    for key, value in counter.items():
        out[coarse(key)] += value
    return out


@dataclass(slots=True)
class Totals:
    sessions: int = 0
    requests: int = 0
    tool_calls: int = 0
    result_bytes: int = 0
    exposure: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compactions: int = 0
    avoidable_requests: int = 0
    avoidable_tokens: int = 0


def totals(rows: list[SessionReport]) -> Totals:
    return Totals(
        sessions=len(rows),
        requests=sum(r.responses for r in rows),
        tool_calls=sum(r.tool_calls for r in rows),
        result_bytes=sum(r.result_bytes for r in rows),
        exposure=sum(r.exposure_bytes for r in rows),
        input_tokens=sum(r.usage.billed for r in rows),
        output_tokens=sum(r.usage.output for r in rows),
        compactions=sum(r.compactions for r in rows),
        avoidable_requests=sum(r.avoidable_requests for r in rows),
        avoidable_tokens=sum(r.avoidable_tokens for r in rows),
    )


def human_bytes(value: int) -> str:
    """Decimal units, so the label matches the arithmetic (GB = 1e9)."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000:
            return f"{size:.1f}{unit}"
        size /= 1000
    return f"{size:.1f}PB"


def human_tokens(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1e9:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1e6:.0f}M"
    return f"{value / 1e3:.0f}k"


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def bar(percent: float, width: int = 28) -> str:
    filled = int(round(percent / 100.0 * width))
    return "#" * min(filled, width) + "." * max(0, width - filled)


def aggregate_signatures(rows: Iterable[SessionReport]) -> Counter[str]:
    out: Counter[str] = Counter()
    for row in rows:
        for sig, data in row.signatures.items():
            calls = data.get("calls") if isinstance(data, dict) else None
            if isinstance(calls, int):
                out[sig] += calls
    return out


def _table(
    title: str,
    counter: Counter[str],
    total: int,
    top: int,
    note: str = "",
    extra: Counter[str] | None = None,
    extra_label: str = "",
) -> list[str]:
    lines = [f"\n## {title}"]
    if note:
        lines.append(f"   ({note})")
    lines.append(f"   {'category':<22} {'share':>7} {'value':>13} {'bar':<28} {extra_label}")
    for key, value in counter.most_common(top):
        share = pct(value, total)
        tail = f"{extra.get(key, 0):,}" if extra is not None else ""
        lines.append(f"   {key:<22} {share:6.1f}% {value:>13,} {bar(share)} {tail}")
    return lines


def render_report(rows: list[SessionReport], top: int = 15, label: str = "") -> str:
    """Render the full text report for a set of session rows."""
    if not rows:
        return "no rows"
    calls = merge(rows, "calls_by_cat")
    byte_counts = merge(rows, "bytes_by_cat")
    exposure = merge(rows, "exposure_by_cat")
    calls_c = rollup(calls)
    bytes_c = rollup(byte_counts)
    exposure_c = rollup(exposure)
    tool_calls = merge(rows, "calls_by_tool")
    tool_exposure = merge(rows, "exposure_by_tool")

    total_usage = Usage()
    for row in rows:
        total_usage.input += row.usage.input
        total_usage.cache_read += row.usage.cache_read
        total_usage.cache_write += row.usage.cache_write
        total_usage.output += row.usage.output

    total = totals(rows)
    total_exposure = sum(exposure_c.values())
    all_input = total.input_tokens

    lines: list[str] = []
    lines.append(f"# llm-audit - {len(rows)} sessions" + (f" [{label}]" if label else ""))
    lines.append(f"\nrequests (model calls) : {total.requests:,}")
    lines.append(f"output tokens          : {total.output_tokens:,}")
    lines.append(f"tool calls             : {total.tool_calls:,}")
    lines.append(f"tool-result bytes      : {human_bytes(total.result_bytes)}")
    lines.append(
        f"context exposure       : {human_bytes(total.exposure)} (bytes x requests that follow)"
    )
    lines.append(
        f"amplification          : "
        f"{total.exposure / max(total.result_bytes, 1):,.0f}x average re-send"
    )
    lines.append(
        f"billed input (cache)   : {total_usage.cache_read / 1e9:.2f}B cache-read tok, "
        f"{total_usage.cache_write / 1e9:.2f}B cache-write tok"
    )
    lines.append(f"compactions            : {total.compactions:,}")
    lines.append(
        f"\nAVOIDABLE REQUESTS     : {total.avoidable_requests:,} of {total.requests:,} "
        f"({pct(total.avoidable_requests, total.requests):.1f}%) carried no new information"
    )
    lines.append(
        f"  their context cost   : {human_tokens(total.avoidable_tokens)} input tokens"
        f"  ({pct(total.avoidable_tokens, all_input):.1f}% of all input tokens)"
    )

    lines += _table(
        "CONTEXT EXPOSURE - who eats the window",
        exposure_c,
        sum(exposure_c.values()),
        top,
        note="bytes x number of later requests in the session",
        extra=bytes_c,
        extra_label="raw bytes",
    )
    lines += _table(
        "raw result bytes", bytes_c, sum(bytes_c.values()), top, extra=calls_c, extra_label="calls"
    )
    lines += _table(
        "call counts", calls_c, sum(calls_c.values()), top, extra=bytes_c, extra_label="bytes"
    )

    lines.append("\n## subsystem detail (owner:subcommand)")
    lines.append(f"   {'sector':<26} {'share':>7} {'exposure':>13} {'calls':>8} {'bytes':>12}")
    for key, value in exposure.most_common(top):
        lines.append(
            f"   {key:<26} {pct(value, total_exposure):6.1f}% {human_bytes(value):>13} "
            f"{calls.get(key, 0):>8,} {human_bytes(byte_counts.get(key, 0)):>12}"
        )

    lines.append("\n## by concrete tool (the harness API, not the shell)")
    lines.append(f"   {'tool':<16} {'calls':>8} {'exposure':>13} {'share':>7}")
    for key, value in tool_exposure.most_common(top):
        lines.append(
            f"   {key:<16} {tool_calls.get(key, 0):>8,} {human_bytes(value):>13} "
            f"{pct(value, total_exposure):6.1f}%"
        )

    dup_bytes = sum(r.dup_waste_bytes for r in rows)
    dup_exposure = sum(r.dup_waste_exposure for r in rows)
    replayed = sum(r.replayed_out_bytes for r in rows)
    norm_exposure = sum(r.norm_repeat_exposure for r in rows)
    norm_calls = sum(r.norm_repeat_calls for r in rows)
    empty_total = sum(r.empty_exposure for r in rows)

    lines.append("\n## waste ledger")
    lines.append(f"   duplicate-call bytes (2nd..Nth copy)   : {human_bytes(dup_bytes)}")
    lines.append(
        f"   duplicate-call exposure                : {human_bytes(dup_exposure)}"
        f"  ({pct(dup_exposure, total_exposure):.1f}% of all exposure)"
    )
    lines.append(
        f"   replayed identical outputs             : {human_bytes(replayed)}"
        f"  ({pct(replayed, total.result_bytes):.1f}% of all result bytes)"
    )
    lines.append(
        f"   NO-NEW-INFORMATION repeats             : {norm_calls:,} results, "
        f"{human_bytes(norm_exposure)} exposure"
        f"  ({pct(norm_exposure, total_exposure):.1f}% of all exposure)"
    )
    lines.append(
        f"   EMPTY results (zero information)       : "
        f"{sum(r.empty_calls for r in rows):,} results, {human_bytes(empty_total)} exposure"
    )
    empty_cats: Counter[str] = Counter()
    for row in rows:
        for key, value in row.empty_by_cat.items():
            empty_cats[key] += value
    if empty_cats:
        lines.append("   -- empty results by subsystem:")
        for key, value in empty_cats.most_common(8):
            share = f"{pct(value, max(empty_total, 1)):5.1f}%"
            lines.append(f"      {share}  {human_bytes(value):>10}  {key}")

    return "\n".join(lines)


def top_sessions(rows: list[SessionReport], count: int = 10) -> list[SessionReport]:
    return sorted(rows, key=lambda r: -r.exposure_bytes)[:count]


def rank_dict(rows: list[SessionReport], field: str, top: int) -> list[tuple[str, int]]:
    return merge(rows, field).most_common(top)


def as_rows(payload: list[dict[str, Any]]) -> list[SessionReport]:
    from .transcript import from_dict

    return [from_dict(item) for item in payload]
