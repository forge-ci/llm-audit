"""Find transcripts, scan them in parallel, and load the rows back.

A Claude-harness state directory holds one directory per working directory,
named by slugging the path (``-home-me-dev-x``). Each such directory holds the
main session transcripts at its top level, and each session's subagent
transcripts under ``<session>/subagents/``.

Subagent transcripts are separately billed context, so they are discovered too
and tagged by kind. A subagent's file is sidechain from top to bottom, which is
why the kind has to reach :func:`llm_audit.transcript.read_session`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .transcript import SessionReport, analyze_session, read_session

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# The repositories this audit was built for. A tool is installed in a new
# checkout every so often, so sessions split across slug variants.
TOOL_CORPORA: dict[str, tuple[str, ...]] = {
    "clud": ("-home-niteris-dev-clud", "-home-niteris-dev-clud2", "-home-niteris-dev-clud3"),
    "bosn": ("-home-niteris-dev-bosn",),
    "soldr": ("-home-niteris-dev-soldr", "-home-niteris-dev-soldr2"),
}
TOOLS: tuple[tuple[str, str], ...] = (
    ("clud", "~/dev/clud"),
    ("soldr", "~/dev/soldr"),
    ("bosn", "~/dev/bosn"),
)


@dataclass(frozen=True, slots=True)
class Job:
    """One transcript to scan, and the time window it is scanned for."""

    path: Path
    corpus: str
    kind: str
    since: datetime | None = None


def all_project_dirs() -> list[Path]:
    root = CLAUDE_PROJECTS
    if not root.is_dir():
        return []
    return [entry for entry in sorted(root.iterdir()) if entry.is_dir()]


def dirs_for(name: str) -> list[Path]:
    """Transcript directories for a named corpus."""
    if name in ("fleet", "all"):
        return all_project_dirs()
    if name == "tools":
        names = [slug for slugs in TOOL_CORPORA.values() for slug in slugs]
    else:
        names = list(TOOL_CORPORA.get(name, (name,)))
    return [p for p in (CLAUDE_PROJECTS / n for n in names) if p.is_dir()]


def dirs_for_cwd(cwd: Path) -> list[Path]:
    """The transcript directory for one working directory."""
    # the slug is derived from the working directory's text; it names a state
    # directory rather than addressing one, so it is built as a string.
    slug = "-" + str(cwd).strip("/").replace("/", "-")
    path = CLAUDE_PROJECTS / slug
    if path.is_dir():
        return [path]
    tail = slug.split("-")[-1]
    return [d for d in all_project_dirs() if tail in d.name]


def _recent_enough(path: Path, since: datetime | None) -> bool:
    """Can this file hold a record at or after ``since``?

    A record cannot be newer than the file holding it, so a file whose mtime is
    older than the cutoff cannot contain an in-window record. Skipping it here
    means a 24 h scan parses ~93 transcripts instead of 2,298.
    """
    if since is None:
        return True
    try:
        return path.stat().st_mtime >= since.timestamp()
    except OSError:
        return True


def job_kind(path: Path) -> str:
    """Classify a transcript by its position in the state directory.

    Derived from the absolute path rather than from what the caller passed in,
    because the same tree can be scanned from the projects root or from one
    project directory. ``<projects>/<slug>/<file>.jsonl`` is a main session;
    anything under a ``subagents`` directory belongs to an agent.
    """
    if "subagents" in path.parts:
        return "subagent"
    try:
        relative = path.relative_to(CLAUDE_PROJECTS)
    except ValueError:
        return "nested"
    return "main" if len(relative.parts) == 2 else "nested"


def corpus_for(path: Path, fallback: str) -> str:
    """The project slug a transcript belongs to, whatever root was scanned."""
    try:
        relative = path.relative_to(CLAUDE_PROJECTS)
    except ValueError:
        return fallback
    return relative.parts[0] if relative.parts else fallback


def _jobs_in_dir(path: Path, recursive: bool, since: datetime | None) -> list[Job]:
    """Transcripts under one project directory: main first, then nested."""
    fallback = path.name
    jobs = [
        Job(found, corpus_for(found, fallback), job_kind(found), since)
        for found in sorted(path.glob("*.jsonl"))
        if _recent_enough(found, since)
    ]
    if not recursive:
        return jobs
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        if root_path == path:
            continue
        for name in sorted(files):
            if not name.endswith(".jsonl"):
                continue
            found = root_path / name
            if _recent_enough(found, since):
                jobs.append(Job(found, corpus_for(found, fallback), job_kind(found), since))
    return jobs


def discover(
    paths: Sequence[Path], recursive: bool = True, since: datetime | None = None
) -> list[Job]:
    """Find every transcript under the given paths."""
    jobs: list[Job] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_file():
            if _recent_enough(path, since):
                jobs.append(Job(path, corpus_for(path, path.parent.name), job_kind(path), since))
        elif path.is_dir():
            jobs.extend(_jobs_in_dir(path, recursive, since))
    return jobs


def _scan_one(job: Job) -> SessionReport | None:
    try:
        read = read_session(job.path, job.kind, job.since)
        return analyze_session(read, job.path, job.corpus, job.kind)
    except (OSError, ValueError, AssertionError, KeyError):
        # a corrupt or half-written transcript must not kill the sweep
        return None


def scan(jobs: list[Job], workers: int | None = None) -> Iterator[SessionReport]:
    """Scan jobs in parallel, yielding reports. Failures are skipped."""
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for report in pool.map(_scan_one, jobs, chunksize=2):
            if report is not None:
                yield report


def write_rows(jobs: list[Job], out: Path, workers: int | None = None) -> int:
    """Scan and write one JSON row per session. Returns the number written."""
    out.resolve().parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as handle:
        for report in scan(jobs, workers):
            handle.write(report.to_json() + "\n")
            count += 1
    return count


def load_rows(path: Path) -> list[SessionReport]:
    """Load rows written by :func:`write_rows`."""
    rows: list[SessionReport] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(SessionReport.from_json(line))
    return rows


def by_kind(rows: list[SessionReport], kind: str) -> list[SessionReport]:
    if kind == "all":
        return rows
    return [r for r in rows if r.kind == kind]


def pretty_corpus(corpus: str) -> str:
    """``-home-niteris-dev-obs-rust`` -> ``obs-rust``."""
    text = corpus.lstrip("-")
    for prefix in ("home-niteris-dev-", "home-niteris-"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


__all__ = [
    "CLAUDE_PROJECTS",
    "Job",
    "TOOLS",
    "TOOL_CORPORA",
    "all_project_dirs",
    "by_kind",
    "dirs_for",
    "dirs_for_cwd",
    "discover",
    "load_rows",
    "pretty_corpus",
    "scan",
    "write_rows",
]
