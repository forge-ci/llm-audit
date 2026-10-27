# llm-audit

Attribute agent context spend to the tools that caused it, and find the
repetition that was never necessary.

It reads Claude-harness session transcripts — the JSONL that `clud`, Codex-via-Claude
and everything else on this machine write to `~/.claude/projects/` — and answers
three questions:

1. **Which tool owns the context?** A ranked partition by subsystem
   (`gh:issue`, `gh:run`, `sed`, `grep`, …), not a pooled "bash".
2. **How much does it actually cost?** Not bytes — *exposure*.
3. **What is unnecessary repetition?** Duplicate calls, replayed outputs,
   uninformative polls, and re-reads of a file nothing touched.

Stdlib only. No network. No writes outside the output path.

## Why not raw byte totals

A tool result is not paid for once. It stays in the conversation until
compaction, so every later model request re-sends it as cached input:

```
exposure(result) = result_bytes x (model requests that come after it)
```

A 20 KB result that arrives early in a long session is billed hundreds of times;
a 60 KB result at the very end is billed once. Ranking by bytes therefore ranks
the wrong things.

The second metric is the **avoidable request**: one whose every tool result
carried no new information (empty, or the same output with only timestamps and
ids changed). It re-sent the whole conversation to discover that nothing moved,
and its own `usage` row measures the bill.

## Install

```bash
uv sync                       # dev environment (basedpyright, pytest, ruff)
uv tool install .             # or: uv tool install --force .
llm-audit --help
```

## Use

```bash
# scan a repo's own sessions (main + nested subagent transcripts)
llm-audit scan ~/.claude/projects/-home-niteris-dev-clud --out reports/clud.jsonl

# the ranked report: exposure by subsystem, waste ledger, top repeated calls
llm-audit report reports/clud.jsonl --kind main --top 18

# was the second read of a file justified by an intervening edit?
llm-audit rereads --corpus clud

# how much of the gh traffic is a poll that learned nothing?
llm-audit polls --corpus clud

# does exposure land early (bulk reading) or late?
llm-audit when --corpus clud

# threshold gate + executive summary
llm-audit health reports/clud.jsonl --rereads --summary reports/exec-summary.md
```

`--corpus` accepts a known corpus name (`clud`, `bosn`, `soldr`, `tools`,
`fleet`) or `--dirs` for explicit transcript directories, or `--cwd <path>` for
one working directory.

## Reading the output

- **exposure** ranks by what the bill responds to. Start here.
- **subsystem detail** is `owner:subcommand`, so `gh:issue` and `gh:run` are
  separate rows rather than one pooled "gh".
- **waste ledger** holds four independent signals: duplicate identical calls,
  byte-identical outputs, no-new-information repeats after normalisation, and
  empty results.
- **avoidable requests** is the cost of all of the above together, in input
  tokens, against the sessions' own recorded usage.

## The gate

`llm-audit health` prints a **deterministic, coarse** verdict block so a cron
`monitor` can diff it:

```
verdict: ALERT
avoidable_pct: 9
top_category: file 40
reread_pct: 50
stale_installs: clud,soldr
rule: top_category_pct | file owns 40.1% of all exposure
```

Integers only, no timestamps, no volatile counts: the corpus grows every day, so
anything finer would read as "changed" on every tick and the gate would never
stay quiet. `verdict: OK` means nothing tripped a threshold.

## Design notes

- **Sidechain transcripts.** A main session's file can embed sidechain records
  belonging to a subagent, but a subagent's own file is sidechain from top to
  bottom. Skipping sidechain rows unconditionally zeroes every subagent
  transcript, which is why `read_session` takes the file's own kind.
- **The classifier is heuristic and tested.** It handles assignment prefixes,
  `$(...)` command substitutions, `cd x && cmd`, and `for`/`while`/`until`
  bodies. Anything it cannot attribute lands in an explicit `bash:<name>` bucket
  and is reported by `report`, never silently pooled.
- **File identity uses the last three path components**, so an absolute `Edit`
  path and a relative `sed` path resolve to the same file without collapsing
  every `lib.rs` in a workspace together. A bare filename is excluded rather
  than guessed at.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -q
```
