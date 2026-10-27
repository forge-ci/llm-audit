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

import glob
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
    """One transcript to scan."""

    path: str
    corpus: str
    kind: str


def all_project_dirs() -> list[str]:
    root = CLAUDE_PROJECTS
    if not root.is_dir():
        return []
    return [str(root / name) for name in sorted(os.listdir(root)) if (root / name).is_dir()]


def dirs_for(name: str) -> list[str]:
    """Transcript directories for a named corpus."""
    if name in ("fleet", "all"):
        return all_project_dirs()
    if name == "tools":
        names = [slug for slugs in TOOL_CORPORA.values() for slug in slugs]
    else:
        names = list(TOOL_CORPORA.get(name, (name,)))
    return [p for p in (str(CLAUDE_PROJECTS / n) for n in names) if os.path.isdir(p)]


def dirs_for_cwd(cwd: str) -> list[str]:
    """The transcript directory for one working directory."""
    slug = "-" + cwd.strip("/").replace("/", "-")
    path = CLAUDE_PROJECTS / slug
    if path.is_dir():
        return [str(path)]
    tail = slug.split("-")[-1]
    return [d for d in all_project_dirs() if tail in os.path.basename(d)]


def discover(paths: list[str], recursive: bool = True) -> list[Job]:
    """Find every transcript under the given directories."""
    jobs: list[Job] = []
    for raw in paths:
        path = os.path.expanduser(raw)
        if os.path.isfile(path):
            jobs.append(Job(path, os.path.basename(os.path.dirname(path)), "main"))
            continue
        if not os.path.isdir(path):
            continue
        corpus = os.path.basename(os.path.normpath(path))
        for found in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
            jobs.append(Job(found, corpus, "main"))
        if not recursive:
            continue
        for root, _dirs, files in os.walk(path):
            if root == path:
                continue
            is_subagent = "subagents" in Path(root).parts
            kind = "subagent" if is_subagent else "nested"
            for name in sorted(files):
                if name.endswith(".jsonl"):
                    jobs.append(Job(os.path.join(root, name), corpus, kind))
    return jobs


def _scan_one(job: Job) -> SessionReport | None:
    try:
        return analyze_session(read_session(job.path, job.kind), job.path, job.corpus, job.kind)
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


def write_rows(jobs: list[Job], out: str, workers: int | None = None) -> int:
    """Scan and write one JSON row per session. Returns the number written."""
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(out, "w") as handle:
        for report in scan(jobs, workers):
            handle.write(report.to_json() + "\n")
            count += 1
    return count


def load_rows(path: str) -> list[SessionReport]:
    """Load rows written by :func:`write_rows`."""
    rows: list[SessionReport] = []
    with open(path, errors="replace") as handle:
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
