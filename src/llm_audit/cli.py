"""Command-line entry points."""

from __future__ import annotations

import argparse
import os
import sys

from .aggregate import render_report
from .analysis import (
    analyze_polls,
    analyze_rereads,
    analyze_when,
    render_polls,
    render_rereads,
    render_when,
)
from .corpus import dirs_for, dirs_for_cwd, discover, load_rows, write_rows
from .health import (
    evaluate,
    exec_summary,
    load_state,
    render_verdict,
    save_state,
)
from .transcript import SessionReport

DEFAULT_STATE = "health-state.json"


def _jobs_from(args: argparse.Namespace) -> list:
    if getattr(args, "dirs", None):
        paths = list(args.dirs)
    elif getattr(args, "cwd", None):
        paths = dirs_for_cwd(args.cwd)
    else:
        paths = dirs_for(getattr(args, "corpus", None) or "tools")
    return discover(paths)


def _scoped_rows(args: argparse.Namespace) -> list[SessionReport]:
    rows = load_rows(args.rows)
    if getattr(args, "corpus", None):
        rows = [r for r in rows if r.corpus == args.corpus]
    kind = getattr(args, "kind", "all")
    if kind != "all":
        rows = [r for r in rows if r.kind == kind]
    return rows


def cmd_scan(args: argparse.Namespace) -> int:
    jobs = discover(args.paths)
    if not jobs:
        print("no transcripts found", file=sys.stderr)
        return 1
    count = write_rows(jobs, args.out, args.workers)
    print(f"scanned {count} transcripts -> {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = _scoped_rows(args)
    if not rows:
        print("no rows for that scope", file=sys.stderr)
        return 1
    print(render_report(rows, top=args.top, label=args.corpus or ""))
    return 0


def cmd_rereads(args: argparse.Namespace) -> int:
    print(render_rereads(analyze_rereads(_jobs_from(args)), top=args.top))
    return 0


def cmd_polls(args: argparse.Namespace) -> int:
    print(render_polls(analyze_polls(_jobs_from(args)), top=args.top))
    return 0


def cmd_when(args: argparse.Namespace) -> int:
    print(render_when(analyze_when(_jobs_from(args))))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    all_rows = load_rows(args.rows)
    rows = all_rows
    if args.kind != "all":
        rows = [r for r in rows if r.kind == args.kind]
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    previous = load_state(args.state)
    reread_pct: float | None = None
    if args.rereads:
        verdict = analyze_rereads(_jobs_from(args))
        if verdict.reads:
            reread_pct = 100.0 * verdict.gratuitous / verdict.reads

    gate = evaluate(rows, all_rows, previous, reread_pct)
    print(render_verdict(gate))
    if args.save:
        save_state(args.state, gate)
    if args.summary:
        parent = os.path.dirname(os.path.abspath(args.summary))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.summary, "w") as handle:
            handle.write(exec_summary(rows, gate, all_rows) + "\n")
        if not args.quiet:
            print(f"\nsummary -> {args.summary}", file=sys.stderr)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    rows = _scoped_rows(args)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    gate = evaluate(rows, rows, load_state(args.state))
    text = exec_summary(rows, gate, rows)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"summary -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


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
    scan.set_defaults(func=cmd_scan)

    report = sub.add_parser("report", help="ranked report from scanned rows")
    report.add_argument("rows")
    report.add_argument("--kind", default="main", choices=["main", "subagent", "nested", "all"])
    report.add_argument("--corpus")
    report.add_argument("--top", type=int, default=15)
    report.set_defaults(func=cmd_report)

    for name, func, helptext in (
        ("rereads", cmd_rereads, "was the second read of a file justified?"),
        ("polls", cmd_polls, "how much gh polling learned nothing"),
        ("when", cmd_when, "where in a session exposure accumulates"),
    ):
        node = sub.add_parser(name, help=helptext)
        node.add_argument("--dirs", nargs="*")
        node.add_argument("--corpus", default="tools")
        node.add_argument("--cwd")
        node.add_argument("--top", type=int, default=15)
        node.set_defaults(func=func)

    health = sub.add_parser("health", help="threshold gate and executive summary")
    health.add_argument("rows")
    health.add_argument("--kind", default="main", choices=["main", "subagent", "nested", "all"])
    health.add_argument("--state", default=DEFAULT_STATE)
    health.add_argument("--summary")
    health.add_argument("--save", action="store_true")
    health.add_argument("--rereads", action="store_true", help="also measure re-read share")
    health.add_argument("--dirs", nargs="*")
    health.add_argument("--corpus", default="tools")
    health.add_argument("--cwd")
    health.add_argument("--top", type=int, default=15)
    health.add_argument("--quiet", action="store_true")
    health.set_defaults(func=cmd_health)

    summary = sub.add_parser("summary", help="executive summary from scanned rows")
    summary.add_argument("rows")
    summary.add_argument("--kind", default="all", choices=["main", "subagent", "nested", "all"])
    summary.add_argument("--corpus")
    summary.add_argument("--state", default=DEFAULT_STATE)
    summary.add_argument("--out")
    summary.set_defaults(func=cmd_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
