"""llm-audit - attribute agent context spend to the tools that caused it.

The core metric is exposure::

    exposure(result) = result_bytes x (model requests that come after it)

because a tool result stays in the conversation until compaction, so every
later request re-sends it as cached input. Rank by that, not by raw bytes.
"""

from __future__ import annotations

from .aggregate import human_bytes, human_tokens, pct, render_report, totals
from .analysis import (
    analyze_polls,
    analyze_rereads,
    analyze_when,
    render_polls,
    render_rereads,
    render_when,
)
from .classify import classify_command, coarse, filekey, normalize_output
from .corpus import Job, dirs_for, discover, load_rows, write_rows
from .health import evaluate, exec_summary, installed_versions, render_verdict
from .transcript import (
    Call,
    Result,
    SessionReport,
    Usage,
    analyze_session,
    read_session,
)

__version__ = "0.1.0"

__all__ = [
    "Call",
    "Job",
    "Result",
    "SessionReport",
    "Usage",
    "__version__",
    "analyze_polls",
    "analyze_rereads",
    "analyze_session",
    "analyze_when",
    "classify_command",
    "coarse",
    "dirs_for",
    "discover",
    "evaluate",
    "exec_summary",
    "filekey",
    "human_bytes",
    "human_tokens",
    "installed_versions",
    "load_rows",
    "normalize_output",
    "pct",
    "read_session",
    "render_polls",
    "render_rereads",
    "render_report",
    "render_verdict",
    "render_when",
    "totals",
    "write_rows",
]
