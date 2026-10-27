"""Discovery: main sessions versus subagent transcripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_audit import corpus
from llm_audit.corpus import discover, pretty_corpus


def build_tree(root: Path) -> None:
    slug = root / "-home-me-dev-x"
    slug.mkdir(parents=True)
    (slug / "main-session.jsonl").write_text("{}\n")
    nested = slug / "main-session" / "subagents" / "workflows" / "wf_1"
    nested.mkdir(parents=True)
    (nested / "agent-a.jsonl").write_text("{}\n")
    (slug / "main-session" / "subagents" / "agent-b.jsonl").write_text("{}\n")


def test_subagents_are_discovered_and_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_tree(tmp_path)
    monkeypatch.setattr(corpus, "CLAUDE_PROJECTS", tmp_path)
    jobs = discover([tmp_path / "-home-me-dev-x"])
    kinds = sorted(job.kind for job in jobs)
    assert kinds == ["main", "subagent", "subagent"]
    assert all(job.corpus == "-home-me-dev-x" for job in jobs)


def test_non_recursive_only_finds_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_tree(tmp_path)
    monkeypatch.setattr(corpus, "CLAUDE_PROJECTS", tmp_path)
    jobs = discover([tmp_path / "-home-me-dev-x"], recursive=False)
    assert len(jobs) == 1
    assert jobs[0].kind == "main"


def test_missing_path_yields_nothing(tmp_path: Path) -> None:
    assert discover([tmp_path / "nope"]) == []


def test_scanning_from_the_projects_root_still_finds_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same tree scanned from the root must not relabel main as nested.

    Regression: kind was inferred from the depth below whatever path was
    passed, so scanning ``~/.claude/projects`` marked every main session
    "nested" and a ``--kind main`` report came back empty.
    """
    build_tree(tmp_path)
    monkeypatch.setattr(corpus, "CLAUDE_PROJECTS", tmp_path)

    from_root = corpus.discover([tmp_path])
    from_project = corpus.discover([tmp_path / "-home-me-dev-x"])

    assert sorted(j.kind for j in from_root) == ["main", "subagent", "subagent"]
    assert sorted(j.kind for j in from_project) == ["main", "subagent", "subagent"]
    # and the slug is the corpus either way, so --corpus filtering still works
    assert {j.corpus for j in from_root} == {"-home-me-dev-x"}


def test_pretty_corpus_shortens_the_slug() -> None:
    assert pretty_corpus("-home-niteris-dev-obs-rust") == "obs-rust"
    assert pretty_corpus("-home-niteris-dev-clud") == "clud"
    assert pretty_corpus("other") == "other"
