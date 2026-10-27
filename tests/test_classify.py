"""Classifier tests.

The classifier is the part most likely to be subtly wrong, so every case here
is one that was actually broken at some point.
"""

from __future__ import annotations

import pytest

from llm_audit.classify import classify_command, coarse, filekey, normalize_output

CASES: list[tuple[str, str]] = [
    # syntax highlighting / basic ownership
    ("cd /repo && cargo build --release", "cargo:build"),
    ("FOO=1 gh pr view 22", "gh:pr"),
    ("gh api repos/a/b/actions/runs/1 -q .status", "gh:api"),
    ("git -C /x fetch -q origin", "git:fetch"),
    ("git checkout -b feature", "git:checkout"),
    ("bosn build --release", "bosn:build"),
    ("bosn ci lint", "bosn:ci"),
    # an assignment must not swallow the real command
    ("B=/tmp/x/scratchpad/issue.md; cat > $B <<'EOF'\nhi\nEOF", "cat"),
    ("j=$(gh run view 123 --json jobs); echo $j", "gh:run"),
    ("E=/home/niteris/dev/clud-extern; grep -rn foo $E/src | head", "grep"),
    ("V=$(git rev-parse HEAD) && echo $V", "git:rev-parse"),
    # loop keywords must not hide the command
    ("for f in *; do echo $f; done", "echo"),
    ("until gh run view 1 -R a/b; do sleep 10; done", "gh:run"),
    ("while pgrep -f daemon; do sleep 5; done", "pgrep"),
    ("if [ -d x ]; then ls x; fi", "ls"),
    # builtins and navigation are never the owner
    ("[ -d $s ] || git clone -q https://x $s", "git:clone"),
    ("test -f a && wc -l a", "wc"),
    ("{ cd /tmp; pwd; }", "pwd"),
    ("time cargo test", "cargo:test"),
    # a flag value is not the subcommand
    (
        "(cd x && uvx --from git+https://github.com/z/ci.yml@abc ci-lint local-gate run)",
        "uvx:ci-lint",
    ),
    ("scratchpad/foo.sh arg", "bash:scratchpad"),
    ("", "bash:empty"),
    ("   ", "bash:empty"),
    ("cd /tmp", "bash:empty"),
]


@pytest.mark.parametrize(("command", "expected"), CASES)
def test_classify(command: str, expected: str) -> None:
    assert classify_command(command) == expected


def test_coarse_rolls_up_owner() -> None:
    assert coarse("gh:api") == "gh"
    assert coarse("gh") == "gh"
    assert coarse("bash:scratchpad") == "bash"


def test_signature_normalisation_is_stable() -> None:
    from llm_audit.transcript import normalize_signature

    spaced = normalize_signature("Bash", {"command": "gh api repos/x/1   -q  .status"})
    tight = normalize_signature("Bash", {"command": "gh api repos/x/1 -q .status"})
    assert spaced == tight

    one = normalize_signature("Bash", {"command": "wc -l aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"})
    two = normalize_signature("Bash", {"command": "wc -l bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2"})
    assert one == two  # long hex-ish ids fold to <sha>


def test_normalize_output_ignores_timestamps_not_numbers() -> None:
    first = normalize_output("2026-01-01T00:00:00Z run 12 queued in 1.2s")
    second = normalize_output("2026-06-02T11:22:33Z run 12 queued in 9.5s")
    assert first == second

    # a count changing is new information and must survive normalisation
    assert normalize_output("12 checks passed") != normalize_output("13 checks passed")


def test_filekey_keeps_files_distinct() -> None:
    assert filekey("/home/niteris/dev/clud/crates/x/src/lib.rs") == "x/src/lib.rs"
    assert filekey("crates/x/src/lib.rs") == "x/src/lib.rs"
    # a bare filename has nothing to disambiguate it for us
    assert filekey("lib.rs") is None
    assert filekey("$E/src/main.rs") == "src/main.rs"
