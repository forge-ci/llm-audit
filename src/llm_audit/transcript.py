"""Read Claude-harness transcripts and compute the per-session cost metrics.

A transcript row is one of ``assistant`` / ``user`` / ``system`` / ... The two
that matter here are an assistant row carrying ``tool_use`` blocks and a user
row carrying the matching ``tool_result`` blocks. ``usage`` lives on the
assistant row.

The core metric is **exposure**::

    exposure(result) = result_bytes x (requests that come after it)

A tool result is not paid for once. It stays in the conversation until
compaction, so every later request re-sends it as cached input. Raw byte totals
rank the wrong things: a 20 KB result early in a long session is billed
hundreds of times, a 60 KB result at the end once.

The secondary metric is an **avoidable request**: one whose every tool result
carried no new information. It re-sent the whole conversation to learn nothing,
and its own ``usage`` row measures the bill.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .classify import classify_command, coarse, filekey, normalize_output
from .window import in_window

# An empty result carries no information at all -- it is what every `sleep 60`
# in a poll loop produces -- so it gets a sentinel digest instead of being
# skipped, or it would escape the count entirely.
EMPTY_DIGEST = "<EMPTY>"

_SIG_KEYS = ("file_path", "path", "pattern", "url", "query", "description", "prompt")
_WS = re.compile(r"\s+")
_TMP = re.compile(r"/tmp/[A-Za-z0-9_.-]+")
_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")


@dataclass(slots=True)
class Call:
    """One ``tool_use`` block, classified."""

    tool: str
    category: str
    signature: str
    # command and file_path are captured verbatim from the tool's JSON input, so
    # they stay str: they are shell/command text, not filesystem paths we own.
    command: str
    file_path: str
    tool_input: dict[str, Any]


@dataclass(slots=True)
class Result:
    """One ``tool_result`` block, with the call it answered."""

    call: Call
    size: int
    text: str
    responses_before: int


@dataclass(slots=True)
class Usage:
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    peak_context: int = 0

    @property
    def billed(self) -> int:
        """Input tokens this request actually paid for."""
        return self.input + self.cache_read + self.cache_write


def normalize_signature(tool: str, tool_input: dict[str, Any]) -> str:
    """A stable fingerprint of a call, for duplicate and sequence detection."""
    if tool == "Bash":
        cmd = str(tool_input.get("command") or "")
        cmd = _WS.sub(" ", cmd).strip()
        cmd = _TMP.sub("/tmp/<f>", cmd)
        cmd = _SHA.sub("<sha>", cmd)
        return "Bash::" + cmd[:400]
    bits: list[str] = []
    for key in _SIG_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            bits.append(value[:200])
    return f"{tool}::{'|'.join(bits)}"


def category_for(tool: str, tool_input: dict[str, Any]) -> str:
    """The subsystem that owns a call: shell commands by their binary."""
    if tool == "Bash":
        return classify_command(str(tool_input.get("command") or ""))
    if tool in ("BashOutput", "KillShell"):
        return "bash:control"
    return _TOOL_CATEGORY.get(tool, "tool:" + tool)


_TOOL_CATEGORY: dict[str, str] = {
    "Read": "file:read",
    "Write": "file:write",
    "Edit": "file:edit",
    "NotebookEdit": "file:edit",
    "MultiEdit": "file:edit",
    "Grep": "search:grep",
    "Glob": "search:glob",
    "Task": "subagent",
    "Agent": "subagent",
    "SendMessage": "subagent",
    "WebFetch": "web",
    "WebSearch": "web",
    "TodoWrite": "planning",
    "ExitPlanMode": "planning",
    "ToolSearch": "harness",
    "Skill": "harness",
    "Monitor": "harness",
    "TaskStop": "harness",
    "ListAgents": "harness",
    "Workflow": "harness",
    "ScheduleWakeup": "harness",
    "SendFeedback": "harness",
    "AskUserQuestion": "user",
}


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one transcript's records, skipping blank and unparseable lines."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def result_bytes(block: dict[str, Any]) -> int:
    """Bytes of the text a tool result actually contributes to the context."""
    content = block.get("content")
    if content is None:
        content = block.get("text")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                text = item.get("text")
                total += len(text) if isinstance(text, str) else len(json.dumps(item))
        return total
    if content is None:
        return 0
    return len(json.dumps(content))


def result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


@dataclass(slots=True)
class SessionRead:
    """Everything one transcript pass produces."""

    results: list[Result] = field(default_factory=list)
    responses: int = 0
    usage: list[Usage] = field(default_factory=list)
    cwd: str = ""
    compactions: int = 0
    sidechain_results: int = 0
    sidechain_bytes: int = 0


def _make_call(block: dict[str, Any]) -> Call:
    tool = str(block.get("name") or "?")
    raw_input = block.get("input")
    tool_input = raw_input if isinstance(raw_input, dict) else {}
    return Call(
        tool=tool,
        category=category_for(tool, tool_input),
        signature=normalize_signature(tool, tool_input),
        command=str(tool_input.get("command") or ""),
        file_path=str(tool_input.get("file_path") or tool_input.get("path") or ""),
        tool_input=tool_input,
    )


def _take_assistant(
    record: dict[str, Any],
    blocks: list[dict[str, Any]],
    out: SessionRead,
    pending: dict[str, Call],
    seen_ids: set[str],
) -> None:
    """Count one model request and register the calls it asked for."""
    message = record.get("message")
    if isinstance(message, dict):
        msg_id = message.get("id") or record.get("uuid")
        if isinstance(msg_id, str) and msg_id not in seen_ids:
            seen_ids.add(msg_id)
            out.responses += 1
            usage = message.get("usage")
            if isinstance(usage, dict):
                out.usage.append(_usage_from(usage))
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        block_id = block.get("id")
        if isinstance(block_id, str):
            pending[block_id] = _make_call(block)


def _take_results(
    blocks: list[dict[str, Any]],
    out: SessionRead,
    pending: dict[str, Call],
    is_side: bool,
    drop_unmatched: bool = False,
) -> None:
    """Record the results a request received, attributing each to its call.

    ``drop_unmatched`` is set when a time window is active: a result whose
    ``tool_use`` fell outside the window has no call to attach to, and
    inventing an "unknown" bucket for it would report out-of-window work as
    in-window.
    """
    for block in blocks:
        if block.get("type") != "tool_result":
            continue
        size = result_bytes(block)
        if is_side:
            out.sidechain_results += 1
            out.sidechain_bytes += size
            continue
        use_id = block.get("tool_use_id")
        call = pending.pop(use_id, None) if isinstance(use_id, str) else None
        if call is None:
            if drop_unmatched:
                continue
            call = Call("?", "unknown", "?", "", "", {})
        out.results.append(
            Result(
                call=call,
                size=size,
                text=result_text(block),
                responses_before=out.responses,
            )
        )


def read_session(path: Path, kind: str = "main", since: datetime | None = None) -> SessionRead:
    """One pass over a transcript.

    ``kind`` controls sidechain handling. A main session's file can embed
    sidechain records that belong to a subagent, but a subagent's own file is
    sidechain from top to bottom: skipping there would zero it, and subagent
    context is billed like any other.

    ``since`` scopes the pass to a time window. Only records at or after the
    cutoff are read, so ``responses``, ``usage`` and every exposure figure come
    out as "what this window cost" rather than "what the session cost". A
    session that began three days ago therefore contributes only its in-window
    requests and results.
    """
    skip_sidechain = kind == "main"
    out = SessionRead()
    pending: dict[str, Call] = {}
    seen_ids: set[str] = set()

    for record in iter_records(path):
        rtype = record.get("type")
        if not in_window(record.get("timestamp"), since):
            continue
        if rtype == "system":
            if "compact" in str(record.get("subtype") or ""):
                out.compactions += 1
            continue
        if rtype not in ("assistant", "user"):
            continue
        if not out.cwd:
            out.cwd = str(record.get("cwd") or "")
        blocks = _blocks(record)
        if not blocks:
            continue
        if rtype == "assistant":
            _take_assistant(record, blocks, out, pending, seen_ids)
        else:
            is_side = bool(record.get("isSidechain")) and skip_sidechain
            _take_results(blocks, out, pending, is_side, drop_unmatched=since is not None)
    return out


def _usage_from(raw: dict[str, Any]) -> Usage:
    def as_int(key: str) -> int:
        value = raw.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    cache_read = as_int("cache_read_input_tokens")
    cache_write = as_int("cache_creation_input_tokens")
    tokens_in = as_int("input_tokens")
    return Usage(
        input=tokens_in,
        cache_read=cache_read,
        cache_write=cache_write,
        output=as_int("output_tokens"),
        peak_context=cache_read + cache_write + tokens_in,
    )


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()


def _share(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    denom = max(total, 1)
    return {k: {"v": v, "pct": round(100.0 * v / denom, 2)} for k, v in counter.most_common(40)}


@dataclass(slots=True)
class SessionReport:
    """One session's cost accounting. Serialised to JSONL by :mod:`corpus`."""

    file: str
    session_id: str
    corpus: str
    kind: str
    cwd: str
    responses: int
    compactions: int
    tool_calls: int
    tool_results: int
    result_bytes: int
    exposure_bytes: int
    sidechain_results: int
    sidechain_bytes: int
    usage: Usage
    calls_by_cat: dict[str, int]
    bytes_by_cat: dict[str, int]
    exposure_by_cat: dict[str, int]
    calls_by_tool: dict[str, int]
    bytes_by_tool: dict[str, int]
    exposure_by_tool: dict[str, int]
    order_runs: list[list[Any]]
    share_calls: dict[str, dict[str, float | int]]
    share_bytes: dict[str, dict[str, float | int]]
    share_exposure: dict[str, dict[str, float | int]]
    dup_calls: int
    dup_call_count: int
    dup_waste_bytes: int
    dup_waste_exposure: int
    replayed_outputs: int
    replayed_out_bytes: int
    norm_repeat_calls: int
    norm_repeat_bytes: int
    norm_repeat_exposure: int
    norm_repeat_by_cat: dict[str, int]
    empty_calls: int
    empty_bytes: int
    empty_exposure: int
    empty_by_cat: dict[str, int]
    avoidable_requests: int
    avoidable_tokens: int
    top_dups: list[dict[str, Any]]
    top_biggest: list[dict[str, Any]]
    top_replayed: list[dict[str, Any]]
    norm_repeat_examples: list[dict[str, Any]]
    signatures: dict[str, dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(line: str) -> SessionReport:
        raw: Any = json.loads(line)
        return from_dict(raw)


def _waste_stats(
    sig_calls: Counter[str],
    sig_bytes: Counter[str],
    sig_exposure: Counter[str],
    text_calls: Counter[str],
    text_bytes: Counter[str],
) -> tuple[list[str], int, int, list[str], int]:
    """Duplicate-call and byte-identical-replay waste.

    Bytes are counted as "the copies after the first one", which is the only
    part that is pure waste: the first copy carried the information.
    """
    dup_sigs = [s for s, count in sig_calls.items() if count > 1]
    dup_waste_bytes = sum(sig_bytes[s] - sig_bytes[s] // sig_calls[s] for s in dup_sigs)
    dup_waste_exposure = int(
        sum(sig_exposure[s] * (sig_calls[s] - 1) / sig_calls[s] for s in dup_sigs)
    )
    replayed = [d for d, count in text_calls.items() if count > 1]
    replayed_bytes = sum(text_bytes[d] - text_bytes[d] // text_calls[d] for d in replayed)
    return dup_sigs, dup_waste_bytes, dup_waste_exposure, replayed, replayed_bytes


def _avoidable_requests(read: SessionRead, informative: list[bool]) -> tuple[list[int], int]:
    """Requests whose every tool result carried no new information.

    These re-sent the whole conversation to discover nothing changed, so their
    own ``usage`` row is the bill for them.
    """
    by_response: dict[int, list[bool]] = {}
    for index, result in enumerate(read.results):
        by_response.setdefault(result.responses_before, []).append(informative[index])
    avoidable = [index for index, flags in by_response.items() if not any(flags)]
    tokens = 0
    for index in avoidable:
        # a result is tagged with the responses seen BEFORE it, so the request
        # that asked for it sits at index - 1 in the usage list
        previous = index - 1
        if 0 <= previous < len(read.usage):
            tokens += read.usage[previous].billed
    return avoidable, tokens


def analyze_session(read: SessionRead, file: Path, corpus: str, kind: str) -> SessionReport:
    """Turn one transcript pass into the metric row."""
    total_responses = read.responses
    tool_calls: Counter[str] = Counter()
    cat_calls: Counter[str] = Counter()
    for result in read.results:
        tool_calls[result.call.tool] += 1
        cat_calls[result.call.category] += 1

    # Categories and order come from the calls, not the results, so a call
    # whose result never landed is still counted.
    call_cat: Counter[str] = Counter()
    order_runs: list[list[Any]] = []
    for result in read.results:
        cat = result.call.category
        call_cat[cat] += 1
        if order_runs and order_runs[-1][0] == cat and order_runs[-1][1] < 60:
            order_runs[-1][1] += 1
        else:
            order_runs.append([cat, 1])

    bytes_by_cat: Counter[str] = Counter()
    exposure_by_cat: Counter[str] = Counter()
    calls_by_cat: Counter[str] = Counter()
    bytes_by_tool: Counter[str] = Counter()
    exposure_by_tool: Counter[str] = Counter()
    sig_bytes: Counter[str] = Counter()
    sig_calls: Counter[str] = Counter()
    sig_exposure: Counter[str] = Counter()
    sig_cat: dict[str, str] = {}
    text_bytes: Counter[str] = Counter()
    text_calls: Counter[str] = Counter()
    seen_norm: set[str] = set()
    norm_repeat_by_cat: Counter[str] = Counter()
    empty_by_cat: Counter[str] = Counter()
    informative: list[bool] = []
    biggest: list[tuple[int, str, str, str, int]] = []

    norm_repeat_calls = norm_repeat_bytes = norm_repeat_exposure = 0
    empty_calls = empty_bytes = empty_exposure = 0
    total_result_bytes = 0
    total_exposure = 0

    for result in read.results:
        cat = result.call.category
        size = result.size
        remaining = max(0, total_responses - result.responses_before)
        exposure = size * remaining
        total_result_bytes += size
        total_exposure += exposure
        bytes_by_cat[cat] += size
        exposure_by_cat[cat] += exposure
        calls_by_cat[cat] += 1
        tool = result.call.tool
        bytes_by_tool[tool] += size
        exposure_by_tool[tool] += exposure
        sig = result.call.signature
        sig_bytes[sig] += size
        sig_calls[sig] += 1
        sig_exposure[sig] += exposure
        sig_cat.setdefault(sig, cat)
        text = result.text
        digest = _digest(text) if text else ""
        norm = normalize_output(text)
        digest_norm = _digest(norm) if norm else EMPTY_DIGEST
        if digest:
            text_bytes[digest] += size
            text_calls[digest] += 1
        was_new = digest_norm != EMPTY_DIGEST and digest_norm not in seen_norm
        informative.append(was_new)
        if digest_norm == EMPTY_DIGEST:
            empty_calls += 1
            empty_bytes += size
            empty_exposure += exposure
            empty_by_cat[cat] += exposure
        if digest_norm in seen_norm:
            norm_repeat_calls += 1
            norm_repeat_bytes += size
            norm_repeat_exposure += exposure
            norm_repeat_by_cat[cat] += exposure
        else:
            seen_norm.add(digest_norm)
        biggest.append((size, cat, tool, sig[:180], remaining))

    biggest.sort(reverse=True)

    dup_sigs, dup_waste_bytes, dup_waste_exposure, replayed, replayed_out_bytes = _waste_stats(
        sig_calls, sig_bytes, sig_exposure, text_calls, text_bytes
    )
    avoidable, avoidable_tokens = _avoidable_requests(read, informative)

    return SessionReport(
        file=str(file),
        session_id=file.name.rsplit(".", 1)[0],
        corpus=corpus,
        kind=kind,
        cwd=read.cwd,
        responses=total_responses,
        compactions=read.compactions,
        tool_calls=sum(tool_calls.values()),
        tool_results=len(read.results),
        result_bytes=total_result_bytes,
        exposure_bytes=total_exposure,
        sidechain_results=read.sidechain_results,
        sidechain_bytes=read.sidechain_bytes,
        usage=_sum_usage(read.usage),
        calls_by_cat=dict(call_cat),
        bytes_by_cat=dict(bytes_by_cat),
        exposure_by_cat=dict(exposure_by_cat),
        calls_by_tool=dict(tool_calls),
        bytes_by_tool=dict(bytes_by_tool),
        exposure_by_tool=dict(exposure_by_tool),
        order_runs=order_runs,
        share_calls=_share(cat_calls, sum(cat_calls.values())),
        share_bytes=_share(bytes_by_cat, total_result_bytes),
        share_exposure=_share(exposure_by_cat, total_exposure),
        dup_calls=len(dup_sigs),
        dup_call_count=sum(c for c in sig_calls.values() if c > 1),
        dup_waste_bytes=dup_waste_bytes,
        dup_waste_exposure=dup_waste_exposure,
        replayed_outputs=len(replayed),
        replayed_out_bytes=replayed_out_bytes,
        norm_repeat_calls=norm_repeat_calls,
        norm_repeat_bytes=norm_repeat_bytes,
        norm_repeat_exposure=norm_repeat_exposure,
        norm_repeat_by_cat=dict(norm_repeat_by_cat),
        empty_calls=empty_calls,
        empty_bytes=empty_bytes,
        empty_exposure=empty_exposure,
        empty_by_cat=dict(empty_by_cat),
        avoidable_requests=len(avoidable),
        avoidable_tokens=avoidable_tokens,
        top_dups=[
            {
                "sig": s[:200],
                "cat": sig_cat.get(s, "?"),
                "calls": sig_calls[s],
                "bytes": sig_bytes[s],
                "exposure": sig_exposure[s],
                "redundant_exposure": int(sig_exposure[s] * (sig_calls[s] - 1) / sig_calls[s]),
            }
            for s in sorted(dup_sigs, key=lambda s: -sig_exposure[s])[:10]
        ],
        top_biggest=[
            {"bytes": b, "cat": c, "tool": t, "sig": s, "later": r}
            for b, c, t, s, r in biggest[:10]
        ],
        top_replayed=[
            {"bytes": text_bytes[d], "calls": text_calls[d]}
            for d in sorted(replayed, key=lambda d: -text_bytes[d])[:10]
        ],
        norm_repeat_examples=[
            {"cat": c, "exposure": v} for c, v in norm_repeat_by_cat.most_common(12)
        ],
        signatures={
            s: {
                "calls": sig_calls[s],
                "bytes": sig_bytes[s],
                "exposure": sig_exposure[s],
                "cat": sig_cat.get(s, "?"),
            }
            for s in sig_calls
        },
    )


def _sum_usage(rows: list[Usage]) -> Usage:
    return Usage(
        input=sum(u.input for u in rows),
        cache_read=sum(u.cache_read for u in rows),
        cache_write=sum(u.cache_write for u in rows),
        output=sum(u.output for u in rows),
        peak_context=max((u.peak_context for u in rows), default=0),
    )


def from_dict(raw: dict[str, Any]) -> SessionReport:
    """Rebuild a report from JSONL, tolerant of fields a newer writer added."""

    def d_int(key: str) -> int:
        value = raw.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    def d_dict(key: str) -> dict[str, int]:
        value = raw.get(key)
        if not isinstance(value, dict):
            return {}
        return {str(k): int(v) for k, v in value.items() if isinstance(v, (int, float))}

    def d_any(key: str) -> dict[str, Any]:
        value = raw.get(key)
        return value if isinstance(value, dict) else {}

    def d_list(key: str) -> list[Any]:
        value = raw.get(key)
        return value if isinstance(value, list) else []

    usage_raw = d_any("usage")
    usage = Usage(
        input=int(usage_raw.get("input") or 0),
        cache_read=int(usage_raw.get("cache_read") or 0),
        cache_write=int(usage_raw.get("cache_write") or 0),
        output=int(usage_raw.get("output") or 0),
        peak_context=int(usage_raw.get("peak_context") or 0),
    )
    return SessionReport(
        file=str(raw.get("file") or ""),
        session_id=str(raw.get("session") or raw.get("session_id") or ""),
        corpus=str(raw.get("corpus") or ""),
        kind=str(raw.get("kind") or "main"),
        cwd=str(raw.get("cwd") or ""),
        responses=d_int("responses"),
        compactions=d_int("compactions"),
        tool_calls=d_int("tool_calls"),
        tool_results=d_int("tool_results"),
        result_bytes=d_int("result_bytes"),
        exposure_bytes=d_int("exposure_bytes"),
        sidechain_results=d_int("sidechain_results"),
        sidechain_bytes=d_int("sidechain_bytes"),
        usage=usage,
        calls_by_cat=d_dict("calls_by_cat"),
        bytes_by_cat=d_dict("bytes_by_cat"),
        exposure_by_cat=d_dict("exposure_by_cat"),
        calls_by_tool=d_dict("calls_by_tool"),
        bytes_by_tool=d_dict("bytes_by_tool"),
        exposure_by_tool=d_dict("exposure_by_tool"),
        order_runs=d_list("order_runs"),
        share_calls=d_any("share_calls"),
        share_bytes=d_any("share_bytes"),
        share_exposure=d_any("share_exposure"),
        dup_calls=d_int("dup_calls"),
        dup_call_count=d_int("dup_call_count"),
        dup_waste_bytes=d_int("dup_waste_bytes"),
        dup_waste_exposure=d_int("dup_waste_exposure"),
        replayed_outputs=d_int("replayed_outputs"),
        replayed_out_bytes=d_int("replayed_out_bytes"),
        norm_repeat_calls=d_int("norm_repeat_calls"),
        norm_repeat_bytes=d_int("norm_repeat_bytes"),
        norm_repeat_exposure=d_int("norm_repeat_exposure"),
        norm_repeat_by_cat=d_dict("norm_repeat_by_cat"),
        empty_calls=d_int("empty_calls"),
        empty_bytes=d_int("empty_bytes"),
        empty_exposure=d_int("empty_exposure"),
        empty_by_cat=d_dict("empty_by_cat"),
        avoidable_requests=d_int("avoidable_requests"),
        avoidable_tokens=d_int("avoidable_tokens"),
        top_dups=d_list("top_dups"),
        top_biggest=d_list("top_biggest"),
        top_replayed=d_list("top_replayed"),
        norm_repeat_examples=d_list("norm_repeat_examples"),
        signatures=d_any("signatures"),
    )


__all__ = [
    "Call",
    "EMPTY_DIGEST",
    "Result",
    "SessionRead",
    "SessionReport",
    "Usage",
    "analyze_session",
    "category_for",
    "coarse",
    "filekey",
    "from_dict",
    "iter_records",
    "normalize_signature",
    "read_session",
    "result_bytes",
    "result_text",
]
