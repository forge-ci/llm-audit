"""Shared transcript builder for the analysis tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_calls(path: Path, calls: list[tuple[str, dict[str, Any], str]]) -> None:
    """Write a transcript from ``(tool_name, tool_input, result_text)`` triples."""
    lines: list[str] = []
    for index, (tool, tool_input, result) in enumerate(calls):
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "cwd": "/repo",
                    "message": {
                        "id": f"m{index}",
                        "role": "assistant",
                        "model": "test-model",
                        "usage": {"input_tokens": 10, "cache_read_input_tokens": 100},
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"t{index}",
                                "name": tool,
                                "input": tool_input,
                            }
                        ],
                    },
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t{index}",
                                "content": result,
                            }
                        ],
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


def bash(command: str) -> dict[str, Any]:
    return {"command": command}


def edit(path_: str) -> dict[str, Any]:
    return {"file_path": path_, "old_string": "a", "new_string": "b"}
