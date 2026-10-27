"""Three follow-up analyses the aggregate report cannot answer.

``rereads``  was the second read of a file justified by an intervening edit?

``polls``    how much of the gh traffic is a poll that learned nothing?

``when``     where in a session does the exposure accumulate?

All three need the command text and the result text, so they re-read the
transcripts rather than working from the rows file.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .aggregate import bar, human_bytes, pct
from .classify import PATH_RE, classify_command, filekey
from .corpus import Job
from .transcript import SessionRead, read_session

# ---------------------------------------------------------------- re-reads

# Bash forms that write a file.
_BASH_WRITE = re.compile(r"(?:>>?|tee\s+-?a?\s*)\s*([\w./$~{}-]+\.[A-Za-z0-9]{1,8})")
_BASH_SEDI = re.compile(r"sed\s+-i[^\s]*\s+(?:-e\s+)?\S+\s+([\w./$~{}-]+\.[A-Za-z0-9]{1,8})")

READ_TOOLS: frozenset[str] = frozenset(
    {"sed", "cat", "grep", "rg", "head", "tail", "nl", "wc", "diff"}
)


def read_targets(tool: str, command: str, file_path: str) -> list[str]:
    """Files this call reads, if it is a read-ish call."""
    # command/file_path are shell and tool-input text and the results are path
    # strings scraped from that text -- none of them are filesystem paths we own.
    if tool == "Read":
        return [file_path] if file_path else []
    if tool != "Bash" or not command:
        return []
    if classify_command(command).split(":")[0] not in READ_TOOLS:
        return []
    return [m.group(1) for m in PATH_RE.finditer(command)]


@dataclass(slots=True)
class RereadVerdict:
    reads: int = 0
    first_reads: int = 0
    justified: int = 0
    gratuitous: int = 0
    exposure_all: int = 0
    exposure_justified: int = 0
    exposure_gratuitous: int = 0
    worst: list[tuple[str, int, int]] = field(default_factory=list)


EDIT_TOOLS: frozenset[str] = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

Event = tuple[int, str, str, int]  # (responses_before, "read"|"write", filekey, exposure)


def _session_events(read: SessionRead) -> list[Event]:
    """Every read and write in one session, keyed by file and ordered later."""
    events: list[Event] = []
    for result in read.results:
        call = result.call
        exposure = result.size * max(0, read.responses - result.responses_before)
        if call.tool in EDIT_TOOLS:
            key = filekey(call.file_path) if call.file_path else None
            if key:
                events.append((result.responses_before, "write", key, 0))
            continue
        if call.tool == "Bash" and call.command:
            for pattern in (_BASH_SEDI, _BASH_WRITE):
                for match in pattern.finditer(call.command):
                    key = filekey(match.group(1))
                    if key:
                        events.append((result.responses_before, "write", key, 0))
        for target in set(read_targets(call.tool, call.command, call.file_path)):
            key = filekey(target)
            if key:
                events.append((result.responses_before, "read", key, exposure))
    return events


def analyze_rereads(jobs: Iterable[Job]) -> RereadVerdict:
    """Split re-reads into those an edit justified and those nothing did."""
    verdict = RereadVerdict()
    gratuitous_per_file: collections.Counter[str] = collections.Counter()
    rereads_per_file: collections.Counter[str] = collections.Counter()

    for job in jobs:
        read = read_session(job.path, job.kind, job.since)
        if not read.responses:
            continue
        events = _session_events(read)
        first_seen: dict[str, int] = {}
        last_write: dict[str, int] = {}
        for index, kind, key, exposure in sorted(events, key=lambda e: e[0]):
            if kind == "write":
                last_write[key] = index
                continue
            verdict.reads += 1
            verdict.exposure_all += exposure
            previous = first_seen.get(key)
            if previous is None:
                first_seen[key] = index
                verdict.first_reads += 1
                continue
            if last_write.get(key, -1) > previous:
                verdict.justified += 1
                verdict.exposure_justified += exposure
            else:
                verdict.gratuitous += 1
                verdict.exposure_gratuitous += exposure
                gratuitous_per_file[key] += exposure
                rereads_per_file[key] += 1

    verdict.worst = [
        (key, rereads_per_file[key], value) for key, value in gratuitous_per_file.most_common(20)
    ]
    return verdict


def render_rereads(verdict: RereadVerdict, top: int = 15) -> str:
    lines = [
        "# re-read audit: was the second read justified?",
        f"\nread calls with a resolvable path : {verdict.reads:,}",
        f"  first read of that file         : {verdict.first_reads:,}",
        f"  re-read after an edit           : {verdict.justified:,}"
        f"  ({pct(verdict.justified, verdict.reads):.1f}%) - real work",
        f"  re-read with no intervening edit: {verdict.gratuitous:,}"
        f"  ({pct(verdict.gratuitous, verdict.reads):.1f}%) - redundant",
        f"\nexposure of these read calls      : {human_bytes(verdict.exposure_all)}",
        f"  of justified re-reads           : {human_bytes(verdict.exposure_justified)}",
        f"  of gratuitous re-reads          : {human_bytes(verdict.exposure_gratuitous)}"
        f"  ({pct(verdict.exposure_gratuitous, verdict.exposure_all):.1f}% of read exposure)",
        "\n## worst gratuitously re-read files (last 3 path components)",
        f"   {'re-reads':>9} {'exposure':>10}  file",
    ]
    for key, count, value in verdict.worst[:top]:
        lines.append(f"   {count:>9,} {human_bytes(value):>10}  {key}")
    return "\n".join(lines)


# ------------------------------------------------------------------- polls

POLL_RE = re.compile(
    r"\bgh\s+(run\s+(view|list|watch)|pr\s+(checks|view|merge|status)|api\b"
    r"|workflow\s+view|repo\s+view)|pr_merge_watch|gh_broker"
)
TERMINAL_RE = re.compile(
    r"\b(completed|success|failure|cancelled|skipped|timed_out)\b", re.IGNORECASE
)
PENDING_RE = re.compile(
    r"\b(queued|in_progress|pending|requested|waiting|starting)\b", re.IGNORECASE
)


@dataclass(slots=True)
class PollVerdict:
    calls: int = 0
    pending: int = 0
    terminal: int = 0
    exposure_all: int = 0
    exposure_pending: int = 0
    exposure_terminal: int = 0
    by_sector: collections.Counter[str] = field(default_factory=collections.Counter)
    by_sector_exposure: collections.Counter[str] = field(default_factory=collections.Counter)
    streaks: list[tuple[int, Path]] = field(default_factory=list)
    worst_sessions: list[tuple[int, int, Path]] = field(default_factory=list)


def analyze_polls(jobs: Iterable[Job]) -> PollVerdict:
    """Classify poll-shaped calls by what their result actually said.

    A poll that returns ``queued`` or ``in_progress`` is the same answer it gave
    last time. Its byte-exposure is small, but each one is also a whole model
    request that re-sent the conversation, which is why the pending count
    matters more than the pending bytes.
    """
    verdict = PollVerdict()
    for job in jobs:
        read = read_session(job.path, job.kind, job.since)
        if not read.responses:
            continue
        streak = 0
        session_all = 0
        session_pending = 0
        for result in read.results:
            command = result.call.command
            if not command or not POLL_RE.search(command):
                continue
            exposure = result.size * max(0, read.responses - result.responses_before)
            verdict.calls += 1
            verdict.exposure_all += exposure
            session_all += exposure
            sector = result.call.category
            verdict.by_sector[sector] += 1
            verdict.by_sector_exposure[sector] += exposure
            pending = bool(PENDING_RE.search(result.text)) and not TERMINAL_RE.search(result.text)
            if pending:
                verdict.pending += 1
                verdict.exposure_pending += exposure
                session_pending += exposure
                streak += 1
            else:
                verdict.terminal += 1
                verdict.exposure_terminal += exposure
                if streak > 1:
                    verdict.streaks.append((streak, job.path))
                streak = 0
        if streak > 1:
            verdict.streaks.append((streak, job.path))
        if session_all:
            verdict.worst_sessions.append((session_pending, session_all, job.path))

    verdict.streaks.sort(reverse=True)
    verdict.worst_sessions.sort(reverse=True)
    return verdict


def render_polls(verdict: PollVerdict, top: int = 12) -> str:
    lines = [
        "# poll audit: what did the gh traffic actually learn?",
        f"\npoll-shaped gh calls          : {verdict.calls:,}",
        f"  returned queued/in_progress : {verdict.pending:,}"
        f"  ({pct(verdict.pending, verdict.calls):.1f}%) - no new information",
        f"  returned a terminal state   : {verdict.terminal:,}",
        f"\nexposure of all poll calls    : {human_bytes(verdict.exposure_all)}",
        f"  of the pending polls        : {human_bytes(verdict.exposure_pending)}"
        f"  ({pct(verdict.exposure_pending, verdict.exposure_all):.1f}% of poll exposure)",
        f"  of the terminal polls       : {human_bytes(verdict.exposure_terminal)}",
        "\n## poll calls by sector",
        f"   {'calls':>7} {'exposure':>10}  sector",
    ]
    for sector, count in verdict.by_sector.most_common(10):
        exposure = verdict.by_sector_exposure[sector]
        lines.append(f"   {count:>7,} {human_bytes(exposure):>10}  {sector}")
    if verdict.streaks:
        counts = sorted(entry[0] for entry in verdict.streaks)
        lines.append("\n## longest unbroken pending-poll streak per session")
        for streak, path in verdict.streaks[:top]:
            lines.append(f"   {streak:>4} consecutive  {path.name}")
        lines.append(f"\n   sessions with a streak > 1 : {len(counts)}")
        lines.append(f"   median streak              : {counts[len(counts) // 2]}")
        lines.append(f"   p90 streak                 : {counts[int(len(counts) * 0.9)]}")
        lines.append(f"   max streak                 : {counts[-1]}")
    if verdict.worst_sessions:
        lines.append("\n## worst sessions by wasted poll exposure")
        for pending, total, path in verdict.worst_sessions[:top]:
            share = pct(pending, total)
            lines.append(
                f"   {human_bytes(pending):>9} wasted of {human_bytes(total):>9} "
                f"({share:4.1f}%)  {path.name}"
            )
    return "\n".join(lines)


# -------------------------------------------------------------------- when


@dataclass(slots=True)
class WhenVerdict:
    sessions: int = 0
    buckets: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def total(self) -> int:
        return sum(self.buckets)


def analyze_when(jobs: Iterable[Job], min_requests: int = 20) -> WhenVerdict:
    """Bucket exposure by where in the session it landed.

    A result's exposure is proportional to how much conversation follows it, so
    the same bytes cost far more early than late. If exposure concentrates in
    the first quarter, the prescription is structural: do bulk reading late, or
    in a subagent whose context dies with it.
    """
    verdict = WhenVerdict()
    for job in jobs:
        read = read_session(job.path, job.kind, job.since)
        if read.responses < min_requests:
            continue
        verdict.sessions += 1
        for result in read.results:
            if not result.size:
                continue
            exposure = result.size * max(0, read.responses - result.responses_before)
            position = min(3, int(4.0 * result.responses_before / read.responses))
            verdict.buckets[position] += exposure
            verdict.counts[position] += 1
    return verdict


def render_when(verdict: WhenVerdict) -> str:
    labels = ["Q1 (first 25%)", "Q2", "Q3", "Q4 (last 25%)"]
    total = verdict.total
    lines = [
        "# when does the exposure accumulate?",
        f"\n{verdict.sessions:,} sessions with >= 20 requests; "
        "exposure bucketed by request position (quartile)\n",
        f"   {'quartile':<14} {'results':>9} {'exposure':>10} {'share':>7}",
    ]
    for index, label in enumerate(labels):
        share = pct(verdict.buckets[index], total)
        lines.append(
            f"   {label:<14} {verdict.counts[index]:>9,} "
            f"{human_bytes(verdict.buckets[index]):>10} {share:6.1f}% {bar(share)}"
        )
    if verdict.counts[0] and verdict.counts[3]:
        per_first = verdict.buckets[0] / verdict.counts[0]
        per_last = verdict.buckets[3] / verdict.counts[3]
        lines.append(
            f"\n   average exposure per result: Q1 {per_first / 1e3:,.0f}KB vs "
            f"Q4 {per_last / 1e3:,.0f}KB ({per_first / max(per_last, 1):,.1f}x)"
        )
    lines.append(
        f"\n   exposure in the first quartile alone: {human_bytes(verdict.buckets[0])} "
        f"of {human_bytes(total)} ({pct(verdict.buckets[0], total):.1f}%)"
    )
    return "\n".join(lines)


__all__ = [
    "PollVerdict",
    "RereadVerdict",
    "WhenVerdict",
    "analyze_polls",
    "analyze_rereads",
    "analyze_when",
    "read_targets",
    "render_polls",
    "render_rereads",
    "render_when",
]
