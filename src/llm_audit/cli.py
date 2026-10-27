"""Command-line entry points.

``argparse`` produces an untyped ``Namespace``. Nothing past :func:`main` sees
one: each subcommand's arguments are converted into a frozen dataclass
immediately after parsing, so every handler takes a typed object, the
``--since`` string is resolved to a ``datetime`` exactly once, and no code
downstream has to guess with ``getattr`` which fields a command happens to have.

That also makes the handlers testable without a terminal - construct the
options dataclass and call the handler.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from .aggregate import render_report
from .analysis import (
    analyze_polls,
    analyze_rereads,
    analyze_when,
    render_polls,
    render_rereads,
    render_when,
)
from .corpus import Job, dirs_for, dirs_for_cwd, discover, load_rows, write_rows
from .health import evaluate, exec_summary, load_state, render_verdict, save_state
from .transcript import SessionReport
from .window import describe, parse_since

DEFAULT_STATE = "health-state.json"

Kind = Literal["main", "subagent", "nested", "all"]
KINDS: tuple[Kind, ...] = ("main", "subagent", "nested", "all")

SINCE_HELP = (
    "only records from this window: 24h, 90m, 7d, 2w, a bare hour count, "
    "an ISO date (2026-10-26) or an ISO instant. Omit for all time."
)


# --------------------------------------------------------------------- options


@dataclass(frozen=True, slots=True)
class ScanOptions:
    paths: tuple[Path, ...]
    out: Path
    workers: int | None
    since: datetime | None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> ScanOptions:
        return cls(
            paths=tuple(Path(p) for p in ns.paths),
            out=Path(ns.out),
            workers=ns.workers,
            since=parse_since(ns.since),
        )


@dataclass(frozen=True, slots=True)
class ReportOptions:
    rows: Path
    kind: Kind
    corpus: str | None
    top: int

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> ReportOptions:
        return cls(rows=Path(ns.rows), kind=ns.kind, corpus=ns.corpus, top=ns.top)


@dataclass(frozen=True, slots=True)
class ScopeOptions:
    """Where the transcripts to analyse come from."""

    dirs: tuple[Path, ...] | None
    corpus: str
    cwd: Path | None
    since: datetime | None
    top: int

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> ScopeOptions:
        return cls(
            dirs=tuple(Path(d) for d in ns.dirs) if ns.dirs else None,
            corpus=str(ns.corpus),
            cwd=Path(ns.cwd) if ns.cwd else None,
            since=parse_since(ns.since),
            top=ns.top,
        )

    def jobs(self) -> list[Job]:
        if self.dirs:
            paths = list(self.dirs)
        elif self.cwd:
            paths = dirs_for_cwd(self.cwd)
        else:
            paths = dirs_for(self.corpus)
        return discover(paths, since=self.since)


@dataclass(frozen=True, slots=True)
class HealthOptions:
    rows: Path
    kind: Kind
    state: Path
    summary: Path | None
    save: bool
    rereads: bool
    quiet: bool
    scope: ScopeOptions

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> HealthOptions:
        return cls(
            rows=Path(ns.rows),
            kind=ns.kind,
            state=Path(ns.state),
            summary=Path(ns.summary) if ns.summary else None,
            save=bool(ns.save),
            rereads=bool(ns.rereads),
            quiet=bool(ns.quiet),
            scope=ScopeOptions.from_namespace(ns),
        )


@dataclass(frozen=True, slots=True)
class SummaryOptions:
    rows: Path
    kind: Kind
    corpus: str | None
    state: Path
    out: Path | None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> SummaryOptions:
        return cls(
            rows=Path(ns.rows),
            kind=ns.kind,
            corpus=ns.corpus,
            state=Path(ns.state),
            out=Path(ns.out) if ns.out else None,
        )


# -------------------------------------------------------------------- handlers


def _select(
    rows: list[SessionReport], kind: Kind, corpus: str | None = None
) -> list[SessionReport]:
    chosen = [r for r in rows if kind == "all" or r.kind == kind]
    if corpus:
        chosen = [r for r in chosen if r.corpus == corpus]
    return chosen


def cmd_scan(options: ScanOptions) -> int:
    jobs = discover(list(options.paths), since=options.since)
    if not jobs:
        print(f"no transcripts found ({describe(options.since)})", file=sys.stderr)
        return 1
    count = write_rows(jobs, options.out, options.workers)
    print(f"scanned {count} transcripts ({describe(options.since)}) -> {options.out}")
    return 0


def cmd_report(options: ReportOptions) -> int:
    rows = _select(load_rows(options.rows), options.kind, options.corpus)
    if not rows:
        print("no rows for that scope", file=sys.stderr)
        return 1
    print(render_report(rows, top=options.top, label=options.corpus or ""))
    return 0


def cmd_analysis(command: str, options: ScopeOptions) -> int:
    jobs = options.jobs()
    if command == "rereads":
        print(render_rereads(analyze_rereads(jobs), top=options.top))
    elif command == "polls":
        print(render_polls(analyze_polls(jobs), top=options.top))
    elif command == "when":
        print(render_when(analyze_when(jobs)))
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unknown analysis {command}")
    return 0


def cmd_health(options: HealthOptions) -> int:
    all_rows = load_rows(options.rows)
    rows = _select(all_rows, options.kind)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    reread_pct: float | None = None
    if options.rereads:
        verdict = analyze_rereads(options.scope.jobs())
        if verdict.reads:
            reread_pct = 100.0 * verdict.gratuitous / verdict.reads

    gate = evaluate(rows, all_rows, load_state(options.state), reread_pct)
    print(render_verdict(gate))
    if options.save:
        save_state(options.state, gate)
    if options.summary:
        options.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
        options.summary.write_text(exec_summary(rows, gate, all_rows) + "\n")
        if not options.quiet:
            print(f"\nsummary -> {options.summary}", file=sys.stderr)
    return 0


def cmd_summary(options: SummaryOptions) -> int:
    rows = _select(load_rows(options.rows), options.kind, options.corpus)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    gate = evaluate(rows, rows, load_state(options.state))
    text = exec_summary(rows, gate, rows)
    if options.out:
        options.out.write_text(text + "\n")
        print(f"summary -> {options.out}", file=sys.stderr)
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-audit",
        description="Attribute agent context spend to the tools that caused it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan transcripts into JSON rows")
    scan.add_argument("paths", nargs="+")
    scan.add_argument("--out", default="rows.jsonl")
    scan.add_argument("--workers", type=int, default=None)
    scan.add_argument("--since", help=SINCE_HELP)

    report = sub.add_parser("report", help="ranked report from scanned rows")
    report.add_argument("rows")
    report.add_argument("--kind", default="main", choices=list(KINDS))
    report.add_argument("--corpus")
    report.add_argument("--top", type=int, default=15)

    for name, helptext in (
        ("rereads", "was the second read of a file justified?"),
        ("polls", "how much gh polling learned nothing"),
        ("when", "where in a session exposure accumulates"),
    ):
        node = sub.add_parser(name, help=helptext)
        node.add_argument("--dirs", nargs="*")
        node.add_argument("--corpus", default="tools")
        node.add_argument("--cwd")
        node.add_argument("--since", help=SINCE_HELP)
        node.add_argument("--top", type=int, default=15)

    health = sub.add_parser("health", help="threshold gate and executive summary")
    health.add_argument("rows")
    health.add_argument("--kind", default="main", choices=list(KINDS))
    health.add_argument("--state", default=DEFAULT_STATE)
    health.add_argument("--summary")
    health.add_argument("--save", action="store_true")
    health.add_argument("--rereads", action="store_true", help="also measure re-read share")
    health.add_argument("--dirs", nargs="*")
    health.add_argument("--since", help=SINCE_HELP)
    health.add_argument("--corpus", default="tools")
    health.add_argument("--cwd")
    health.add_argument("--top", type=int, default=15)
    health.add_argument("--quiet", action="store_true")

    summary = sub.add_parser("summary", help="executive summary from scanned rows")
    summary.add_argument("rows")
    summary.add_argument("--kind", default="all", choices=list(KINDS))
    summary.add_argument("--corpus")
    summary.add_argument("--state", default=DEFAULT_STATE)
    summary.add_argument("--out")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse once, convert to a typed options object, dispatch."""
    parsed = build_parser().parse_args(argv)
    command = str(parsed.command)
    if command == "scan":
        return cmd_scan(ScanOptions.from_namespace(parsed))
    if command == "report":
        return cmd_report(ReportOptions.from_namespace(parsed))
    if command in ("rereads", "polls", "when"):
        return cmd_analysis(command, ScopeOptions.from_namespace(parsed))
    if command == "health":
        return cmd_health(HealthOptions.from_namespace(parsed))
    if command == "summary":
        return cmd_summary(SummaryOptions.from_namespace(parsed))
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
