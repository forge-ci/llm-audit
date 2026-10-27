"""The scheduled gate and the executive summary.

The gate output must be **deterministic and coarse**. The corpus grows every
day, so exact counts and decimals would read as "changed" on every tick and a
monitor watching this output would wake the agent every single day. Integers
move only when something real moves.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aggregate import human_bytes, human_tokens, pct, totals
from .classify import coarse
from .corpus import TOOLS, pretty_corpus
from .transcript import SessionReport

RULES: dict[str, float] = {
    "avoidable_pct_delta": 1.0,  # points worse than the last run
    "avoidable_pct_abs": 12.0,  # or this bad outright
    "reread_pct": 60.0,  # share of read calls that are re-reads
    "top_category_pct": 30.0,  # one subsystem owning this much exposure
    "new_signature_calls": 20,  # a new repeated invocation this hot
}


@dataclass(slots=True)
class Finding:
    rule: str
    detail: str


@dataclass(slots=True)
class Gate:
    findings: list[Finding] = field(default_factory=list)
    avoidable_pct: float = 0.0
    top_category: str = "-"
    top_category_pct: float = 0.0
    reread_pct: float | None = None
    stale_installs: list[str] = field(default_factory=list)
    known_signatures: list[str] = field(default_factory=list)

    @property
    def is_alert(self) -> bool:
        return bool(self.findings)


def exposure_by_owner(rows: list[SessionReport]) -> collections.Counter[str]:
    out: collections.Counter[str] = collections.Counter()
    for row in rows:
        for key, value in row.exposure_by_cat.items():
            out[coarse(key)] += value
    return out


def signatures(rows: list[SessionReport]) -> collections.Counter[str]:
    out: collections.Counter[str] = collections.Counter()
    for row in rows:
        for sig, data in row.signatures.items():
            calls = data.get("calls") if isinstance(data, dict) else None
            if isinstance(calls, int):
                out[sig] += calls
    return out


def installed_versions() -> dict[str, dict[str, str | None]]:
    """Effective installed version vs in-repo version for each tool.

    "Installed" means the binary first on PATH, not a version found in some
    site-packages directory. Two install roots can disagree, and asking the
    binary is the only answer that matches what a session will run.
    """
    out: dict[str, dict[str, str | None]] = {}
    for name, repo in TOOLS:
        executable = shutil.which(name)
        installed: str | None = None
        if executable:
            try:
                done = subprocess.run(
                    [executable, "--version"], capture_output=True, text=True, timeout=20
                )
                match = re.search(r"(\d+\.\d+\.\d+)", done.stdout or done.stderr or "")
                installed = match.group(1) if match else None
            except (OSError, subprocess.SubprocessError):
                installed = None
        cargo = Path(os.path.expanduser(repo)) / "Cargo.toml"
        repo_version: str | None = None
        if cargo.is_file():
            for line in cargo.read_text(errors="replace").splitlines():
                match = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
                if match:
                    repo_version = match.group(1)
                    break
        out[name] = {"installed": installed, "repo": repo_version}
    return out


def evaluate(
    rows: list[SessionReport],
    all_rows: list[SessionReport],
    previous: dict[str, Any] | None = None,
    reread_pct: float | None = None,
    check_versions: bool = True,
) -> Gate:
    """Apply every threshold. ``rows`` is the scope being judged."""
    previous = previous or {}
    gate = Gate(reread_pct=reread_pct)
    total = totals(rows)
    gate.avoidable_pct = pct(total.avoidable_requests, total.requests)

    owners = exposure_by_owner(rows)
    if owners:
        gate.top_category, top_value = owners.most_common(1)[0]
        gate.top_category_pct = pct(top_value, sum(owners.values()))

    # typed as `object` so the isinstance check narrows it to a number; a plain
    # `Any` from dict.get() keeps `None` in the union and defeats the narrowing
    delta: object = previous.get("avoidable_pct")
    previous_pct = 0.0
    if isinstance(delta, (int, float)):
        previous_pct = float(delta)
    rose = gate.avoidable_pct - previous_pct
    if rose > RULES["avoidable_pct_delta"]:
        gate.findings.append(
            Finding(
                "avoidable_pct_delta",
                f"uninformative requests rose {previous_pct:.1f}% -> "
                f"{gate.avoidable_pct:.1f}% (limit +{RULES['avoidable_pct_delta']})",
            )
        )
    if gate.avoidable_pct > RULES["avoidable_pct_abs"]:
        gate.findings.append(
            Finding(
                "avoidable_pct_abs",
                f"{gate.avoidable_pct:.1f}% of requests are uninformative "
                f"(threshold {RULES['avoidable_pct_abs']}%)",
            )
        )
    if gate.top_category_pct > RULES["top_category_pct"]:
        gate.findings.append(
            Finding(
                "top_category_pct",
                f"{gate.top_category} owns {gate.top_category_pct:.1f}% of all exposure",
            )
        )
    if reread_pct is not None and reread_pct > RULES["reread_pct"]:
        gate.findings.append(
            Finding(
                "reread_pct",
                f"{reread_pct:.1f}% of read calls re-read a file "
                f"(threshold {RULES['reread_pct']}%)",
            )
        )

    sig = signatures([r for r in all_rows if r.kind == "main"])
    # The first run has no baseline at all. A baseline that happens to be empty
    # still counts as a baseline, so test for the key, not for emptiness.
    has_baseline = "known_signatures" in previous
    known = set(previous.get("known_signatures") or [])
    hot_new = [
        (s, c)
        for s, c in sig.most_common(300)
        if s not in known and c >= RULES["new_signature_calls"]
    ]
    if hot_new and has_baseline:  # skip the first run, which has no baseline
        sig_text, count = hot_new[0]
        gate.findings.append(
            Finding(
                "new_signature_calls",
                f"{count} calls of a new invocation: {' '.join(sig_text.split())[:120]}",
            )
        )
    gate.known_signatures = [s for s, _ in sig.most_common(3000)]

    if check_versions:
        drift = installed_versions()
        gate.stale_installs = [
            name
            for name, data in drift.items()
            if data["installed"] and data["installed"] != data["repo"]
        ]
        if gate.stale_installs:
            detail = ", ".join(
                f"{name} {drift[name]['installed']} -> {drift[name]['repo']}"
                for name in gate.stale_installs
            )
            gate.findings.append(Finding("version_drift", "stale installs: " + detail))
    return gate


def render_verdict(gate: Gate) -> str:
    """The deterministic block a cron ``monitor`` gate diffs."""
    lines = [
        f"verdict: {'ALERT' if gate.is_alert else 'OK'}",
        f"avoidable_pct: {round(gate.avoidable_pct)}",
        f"top_category: {gate.top_category} {round(gate.top_category_pct)}",
        f"reread_pct: {'na' if gate.reread_pct is None else round(gate.reread_pct)}",
        "stale_installs: "
        + (",".join(sorted(gate.stale_installs)) if gate.stale_installs else "none"),
    ]
    lines += [f"rule: {f.rule} | {f.detail}" for f in gate.findings]
    return "\n".join(lines)


def load_state(path: str) -> dict[str, Any]:
    try:
        with open(path) as handle:
            loaded: Any = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_state(path: str, gate: Gate) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "avoidable_pct": gate.avoidable_pct,
                "top_category": gate.top_category,
                "known_signatures": gate.known_signatures,
            },
            handle,
        )


def _largest_result(all_rows: list[SessionReport]) -> tuple[int, str, str] | None:
    """The single biggest tool result on the machine, if any."""
    biggest: list[tuple[int, str, str]] = []
    for row in all_rows:
        for entry in row.top_biggest:
            value = entry.get("bytes")
            if isinstance(value, int):
                biggest.append((value, str(entry.get("cat", "?")), pretty_corpus(row.corpus)))
    return max(biggest) if biggest else None


def _worst_session_block(all_rows: list[SessionReport], count: int = 6) -> list[str]:
    worst = sorted(all_rows, key=lambda r: -r.exposure_bytes)[:count]
    if not worst:
        return []
    lines = [
        "",
        "## Worst sessions by exposure",
        "",
        "| exposure | requests | calls | repo |",
        "|---|---|---|---|",
    ]
    for row in worst:
        lines.append(
            f"| {human_bytes(row.exposure_bytes)} | {row.responses:,} | "
            f"{row.tool_calls:,} | {pretty_corpus(row.corpus)} |"
        )
    return lines


def exec_summary(
    rows: list[SessionReport],
    gate: Gate,
    all_rows: list[SessionReport] | None = None,
    per_repo_top: int = 18,
) -> str:
    """The executive summary, as markdown."""
    all_rows = all_rows if all_rows is not None else rows
    total = totals(all_rows)
    lines: list[str] = [
        "# Context health - executive summary",
        "",
        f"**Scope:** {total.sessions:,} transcripts, {total.requests:,} model requests, "
        f"{total.tool_calls:,} tool calls.",
        "",
        f"**{human_tokens(total.input_tokens)} input tokens billed "
        f"({human_tokens(total.output_tokens)} output).** "
        f"{human_bytes(total.result_bytes)} of tool output became "
        f"**{human_bytes(total.exposure)} of re-sent context** "
        f"({total.exposure / max(total.result_bytes, 1):.0f}x).",
        "",
        f"**{total.avoidable_requests:,} requests "
        f"({pct(total.avoidable_requests, total.requests):.1f}%) carried no new "
        f"information** - {human_tokens(total.avoidable_tokens)} input tokens "
        f"({pct(total.avoidable_tokens, total.input_tokens):.1f}% of the bill).",
        "",
        "## Action needed",
    ]
    if gate.findings:
        lines += [f"- **{f.rule}** - {f.detail}" for f in gate.findings]
    else:
        lines.append("- Nothing tripped a threshold this run.")

    lines += [
        "",
        "## Per repository",
        "",
        "| repo | sessions | requests | exposure | avoidable | worst subsystem |",
        "|---|---|---|---|---|---|",
    ]
    grouped: dict[str, list[SessionReport]] = {}
    for row in rows:
        grouped.setdefault(row.corpus, []).append(row)
    ordered = sorted(grouped.items(), key=lambda kv: -sum(r.exposure_bytes for r in kv[1]))
    for corpus, group in ordered[:per_repo_top]:
        group_totals = totals(group)
        owners = exposure_by_owner(group)
        if owners:
            top_owner, top_value = owners.most_common(1)[0]
            worst = f"{top_owner} {pct(top_value, sum(owners.values())):.0f}%"
        else:
            worst = "-"
        lines.append(
            f"| {pretty_corpus(corpus)} | {group_totals.sessions:,} | "
            f"{group_totals.requests:,} | {human_bytes(group_totals.exposure)} | "
            f"{pct(group_totals.avoidable_requests, group_totals.requests):.1f}% | {worst} |"
        )

    lines += ["", "## Machine-wide partition", "", "| subsystem | share of exposure |", "|---|---|"]
    owners_all = exposure_by_owner(all_rows)
    owner_total = sum(owners_all.values())
    for owner, value in owners_all.most_common(10):
        lines.append(f"| {owner} | {pct(value, owner_total):.1f}% |")

    if gate.reread_pct is not None:
        lines += [
            "",
            f"**Re-reading:** {gate.reread_pct:.1f}% of read-ish calls re-read a file "
            "already read in the same session.",
        ]

    largest = _largest_result(all_rows)
    if largest is not None:
        size, category, corpus = largest
        lines += [
            "",
            f"**No single-result blowup:** the largest one result is {human_bytes(size)} "
            f"({category}, {corpus}). The cost is count x persistence, not one giant output.",
        ]

    lines += _worst_session_block(all_rows)

    drift = installed_versions()
    lines += [
        "",
        "## Tool version drift",
        "",
        "| tool | installed (on PATH) | in repo | |",
        "|---|---|---|---|",
    ]
    for name in sorted(drift):
        data = drift[name]
        state = "**stale**" if data["installed"] != data["repo"] else "current"
        lines.append(f"| {name} | {data['installed']} | {data['repo']} | {state} |")

    lines += [
        "",
        "## What to do about it",
        "",
        "1. Route every wait through the fail-fast watcher; forbid bare `sleep` "
        "before a `gh` poll.",
        "2. Stop reading source through bash - repo map first, then a session read "
        "ledger that answers a re-read with `<path> unchanged since request N`.",
        "3. Extend the `gh` read broker to porcelain (`gh issue list`, `gh pr view`, "
        "`gh run view`).",
        "4. Count poll streaks on normalised output, not exact argv, and return "
        "`still <status>, unchanged for N polls`.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "Finding",
    "Gate",
    "RULES",
    "evaluate",
    "exec_summary",
    "exposure_by_owner",
    "installed_versions",
    "load_state",
    "render_verdict",
    "save_state",
    "signatures",
]
