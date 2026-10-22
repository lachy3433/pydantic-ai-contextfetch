"""Live check against the real API (2 credits). Skipped unless CONTEXTFETCH_API_KEY is set."""

from __future__ import annotations

import os

import pytest

from pydantic_ai_contextfetch import ContextFetchToolset

pytestmark = pytest.mark.skipif(not os.environ.get("CONTEXTFETCH_API_KEY"), reason="needs CONTEXTFETCH_API_KEY")


def test_live() -> None:
    tools = ContextFetchToolset(num_results=2).tools
    results = tools["contextfetch_search"].function("python asyncio TaskGroup")
    assert results and results[0]["url"].startswith("http") and results[0]["content"]
    page = tools["contextfetch_read"].function("https://docs.python.org/3/library/asyncio-task.html", query="TaskGroup")
    assert "TaskGroup" in page
