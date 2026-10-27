"""Shell-command classification and text normalisation.

Everything in here is a pure function over strings: no I/O, no state. That is
what makes the classifier testable, and the classifier is the part most likely
to be subtly wrong.
"""

from __future__ import annotations

import os
import re

# Ordered: the first simple command in a shell line that names one of these
# owns the call. Everything not listed falls into a ``bash:<name>`` bucket.
BINARIES: frozenset[str] = frozenset(
    {
        "gh",
        "bosn",
        "clud",
        "soldr",
        "git",
        "cargo",
        "rustc",
        "nextest",
        "uv",
        "uvx",
        "pytest",
        "python3",
        "python",
        "nix",
        "systemctl",
        "docker",
        "npm",
        "pnpm",
        "yarn",
        "node",
        "cmake",
        "make",
        "ninja",
        "clang",
        "g++",
        "cc",
        "ld",
        "strip",
        "jq",
        "curl",
        "wget",
        "act",
        "rg",
        "grep",
        "find",
        "fd",
        "sed",
        "awk",
        "cat",
        "ls",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "xargs",
        "tar",
        "unzip",
        "ssh",
        "scp",
        "rsync",
        "rm",
        "cp",
        "mv",
        "mkdir",
        "touch",
        "chmod",
        "stat",
        "du",
        "df",
        "echo",
        "printf",
        "true",
        "false",
        "date",
        "env",
        "readlink",
        "sha256sum",
        "sha1sum",
        "md5sum",
        "base64",
        "diff",
        "comm",
        "tr",
        "cut",
        "paste",
        "sleep",
        "kill",
        "ps",
        "timeout",
        "tee",
        "which",
        "command",
        "type",
        "pgrep",
        "pkill",
        "uptime",
        "safe-rm",
        "mktemp",
        "realpath",
        "nproc",
        "tput",
        "lsof",
        "ss",
        "fuser",
        "pstree",
        "hostname",
        "whoami",
        "id",
        "free",
        "tree",
        "openssl",
        "xxd",
        "od",
        "strings",
        "nm",
        "objdump",
        "readelf",
        "rustup",
        "rustfmt",
        "meson",
        "go",
        "java",
        "zip",
        "gzip",
        "gunzip",
        "zstd",
        "patch",
        "cmp",
        "claude",
        "codex",
        "kitty",
        "tmux",
        "screen",
        "pip",
        "pipx",
        "less",
        "more",
        "column",
        "rev",
        "seq",
        "yes",
        "sh",
        "bash",
        "zsh",
        "pwd",
        "dirname",
        "basename",
        "truncate",
        "split",
        "fmt",
    }
)

# Shell builtins and navigators that are never the owner of a command:
# ``cd repo && cargo build`` is a cargo call, not a cd call.
IGNORED_LEADERS: frozenset[str] = frozenset(
    {
        "cd",
        "pushd",
        "popd",
        "export",
        "set",
        "unset",
        "source",
        ".",
        ":",
        "true",
        "false",
        "alias",
        "unalias",
        "shopt",
        "ulimit",
        "umask",
        "local",
        "declare",
        "readonly",
        "eval",
        "break",
        "continue",
        "return",
        "then",
        "do",
        "done",
        "fi",
        "if",
        "for",
        "while",
        "until",
        "case",
        "esac",
        "select",
        "time",
        "watch",
        "nohup",
        "test",
        "[",
        "[[",
        "{",
        "!",
    }
)

# Tools worth splitting by subcommand, so a report can say ``gh:issue`` rather
# than pooling every gh invocation into one row.
SUBCOMMANDED: frozenset[str] = frozenset(
    {
        "gh",
        "bosn",
        "clud",
        "git",
        "cargo",
        "docker",
        "npm",
        "pnpm",
        "uv",
        "uvx",
        "nix",
        "systemctl",
        "soldr",
        "pytest",
        "nextest",
        "act",
        "node",
        "cmake",
        "make",
        "python",
        "python3",
        "ssh",
    }
)

# Segment separators that end one simple command and start the next.
_SEG_SPLIT = re.compile(r"&&|\|\||[;|\n]")
# An assignment prefix is stripped only when its value is a simple word:
# ``j=$(gh run view ...)`` must NOT be eaten, or the gh call disappears.
_ENV_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\s(]*\s+")
_LEADING_WRAPPER = re.compile(
    r"^(?:\$\(|\(|time|sudo|nohup|command|exec|env|timeout\s+\S+|"
    r"xargs(?:\s+-\S+)*|find\s+\S+\s+-exec)\s+"
)
# Reserved words that PREFIX a real command on the same segment, as in
# ``do git fetch`` or ``until gh run view``. Strip them and keep looking.
_STRIP_LEAD = re.compile(r"^(?:do|then|else|elif|until|while|if|time|watch|nohup|exec|sudo|!)\b\s*")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_CMDSUBST_OPEN = "$("

_NORM_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_NORM_ID = re.compile(r"\b[0-9a-f]{7,40}\b")
_NORM_DUR = re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h)\b")
_NORM_WS = re.compile(r"[ \t]+")

# A path argument. The extension allowlist keeps code identifiers such as
# ``json.loads`` out of the results.
_EXT = (
    "rs|toml|py|pyi|md|json|json5|yaml|yml|ts|tsx|js|jsx|mjs|cjs|sh|bash|zsh"
    "|proto|txt|lock|html|css|scss|c|h|cc|cpp|cxx|hpp|hh|go|java|kt|rb|pl"
    "|sql|xml|ini|cfg|conf|cmake|gradle|wgsl|glsl|swift|patch|diff"
)
PATH_RE = re.compile(
    r"(?<![\w./-])((?:[\w.@+~$(){}-]+/)*[\w.@+~$(){}-]+\.(?:" + _EXT + r"))"
    r"(?![\w/])"
)


# Command bodies of ``$(...)`` and backtick substitutions inside a segment.
def _cmdsubst_bodies(seg: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(seg)
    while i < n:
        if seg.startswith(_CMDSUBST_OPEN, i):
            depth = 1
            j = i + 2
            while j < n and depth:
                if seg.startswith(_CMDSUBST_OPEN, j):
                    depth += 1
                    j += 1
                elif seg[j] == ")":
                    depth -= 1
                j += 1
            out.append(seg[i + 2 : j - 1])
            i = j
        elif seg[i] == "`":
            j = seg.find("`", i + 1)
            if j < 0:
                break
            out.append(seg[i + 1 : j])
            i = j + 1
        else:
            i += 1
    return out


def _split_segments(cmd: str) -> list[tuple[str, list[str], str]]:
    """Yield ``(raw_head, words, raw_segment)`` for each simple command."""
    out: list[tuple[str, list[str], str]] = []
    for seg in _SEG_SPLIT.split(cmd):
        seg = seg.strip()
        if not seg:
            continue
        for _ in range(6):
            new = _ENV_PREFIX.sub("", seg)
            new = _LEADING_WRAPPER.sub("", new)
            new = _STRIP_LEAD.sub("", new)
            new = new.lstrip("$({ \t")
            if new == seg:
                break
            seg = new
        seg = seg.strip().strip("\"'")
        if not seg:
            continue
        words = [w for w in re.split(r"[\s'\"]+", seg) if w]
        if not words:
            continue
        out.append((words[0], words, seg))
    return out


def _subcommand(words: list[str]) -> str:
    """First word that looks like a subcommand, skipping flags and values."""
    skip_next = False
    for word in words[1:7]:
        if skip_next:
            skip_next = False
            continue
        if word.startswith("-"):
            skip_next = len(word) == 2
            continue
        if "://" in word or word.startswith(("git+", "$")) or word == "." or "/" in word:
            continue
        return word
    return ""


def _owner_of(raw_head: str, words: list[str]) -> str:
    """The category of one simple command, or ``""`` if it owns nothing."""
    # raw_head is a word from a shell command string, not a filesystem path we
    # own, so os.path's text operations on it are the right tool here.
    head = os.path.basename(raw_head)
    if head.endswith((".py", ".sh", ".js")):
        head = head.rsplit(".", 1)[0]
    if head in IGNORED_LEADERS:
        return ""
    if head in BINARIES:
        sub = _subcommand(words)
        if head in SUBCOMMANDED and sub:
            return f"{head}:{sub[:16]}"
        return head
    if "/" in raw_head:
        # a path into the workspace: name it by its directory, which is what a
        # reader recognises (``bash:scratchpad``)
        parent = os.path.basename(os.path.dirname(raw_head)) or "path"
        return "bash:" + parent[:24]
    return "bash:" + head[:24]


def _inner_from_substitution(seg: str, depth: int) -> str:
    """Classify what runs inside ``$(...)`` in a segment, if anything."""
    if depth >= 3:
        return ""
    for body in _cmdsubst_bodies(seg):
        inner = classify_command(body, depth + 1)
        if inner != "bash:empty":
            return inner
    return ""


def classify_command(cmd: str, _depth: int = 0) -> str:
    """Return ``owner`` or ``owner:subcommand`` for a shell command.

    Walks the simple segments in order, skipping navigators, builtins and pure
    assignments. A segment that is only an assignment is still searched for
    command substitutions, so ``j=$(gh run view ...)`` is attributed to gh --
    that substitution's output is what lands in the context.
    """
    if not cmd:
        return "bash:empty"
    for raw_head, words, seg in _split_segments(cmd):
        if not raw_head or raw_head in IGNORED_LEADERS or _ASSIGN.match(raw_head):
            inner = _inner_from_substitution(seg, _depth)
            if inner:
                return inner
            continue
        owner = _owner_of(raw_head, words)
        if owner:
            return owner
    return "bash:empty"


def coarse(category: str) -> str:
    """Roll a fine category (``gh:api``) back up to its owner (``gh``)."""
    return category.split(":", 1)[0]


def normalize_output(text: str) -> str:
    """Collapse a tool result to the part that can carry new information.

    Timestamps, ids and durations change on every call even when nothing has
    happened, so a poll loop produces a "new" result each time. Numbers that
    are not ids or durations are preserved deliberately: "12 checks passed" and
    "13 checks passed" are different facts.
    """
    if not text:
        return ""
    t = _NORM_TS.sub("<TS>", text[:6000])
    t = _NORM_ID.sub("<ID>", t)
    t = _NORM_DUR.sub("<DUR>", t)
    t = _NORM_WS.sub(" ", t)
    return "\n".join(line.rstrip() for line in t.splitlines() if line.strip())


_PREFIXES = ("~", "$E", "$R", "$S", "$A", "$B", "$D", "$P", "$C", "$F", "$Z")


def filekey(path_text: str) -> str | None:
    """A comparable key for a path that may be absolute, relative or ~-prefixed.

    The last three components are used when present, which bridges
    ``/home/me/dev/x/src/lib.rs`` against ``src/lib.rs`` without collapsing
    every ``lib.rs`` in a workspace into one file. A bare filename has nothing
    to disambiguate it, so it returns ``None`` and is excluded rather than
    guessed at.
    """
    # path_text is a path scraped out of tool input or a shell command -- text
    # to compare, not a filesystem path this code owns -- so it stays a str.
    p = path_text.strip().strip("'\"")
    for prefix in _PREFIXES:
        if p.startswith(prefix + "/"):
            p = p[len(prefix) + 1 :]
    parts = [x for x in p.split("/") if x and x not in (".", "..")]
    if len(parts) < 2:
        return None
    return "/".join(parts[-3:])
