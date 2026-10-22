"""Unit tests: the SDK's HTTP call is patched, so nothing touches the network."""

from __future__ import annotations

import json
from typing import Any

import pytest
from contextfetch import ContextFetch, ContextFetchError
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_contextfetch import ContextFetchToolset

RESULTS = [
    {"url": "https://example.com/a", "title": "A", "publishedDate": "2026-01-02", "text": "# A", "prunedContext": "chunk A"},
    {"url": "https://example.com/b", "title": None, "text": "# B", "prunedContext": "chunk B"},
]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.delenv("CONTEXTFETCH_API_KEY", raising=False)

    def request(self: ContextFetch, method: str, path: str, body: Any = None, **_: Any) -> tuple[bytes, dict]:
        calls.append({"method": method, "path": path, "body": body, "api_key": self.api_key})
        if path.startswith("/r/https://missing"):
            raise ContextFetchError("not found", status=404)
        if path.startswith("/r/"):
            return b"# Page", {}
        return json.dumps({"results": RESULTS[: body["numResults"]]}).encode(), {}

    monkeypatch.setattr(ContextFetch, "_request", request)
    return calls


def _call(toolset: ContextFetchToolset, tool: str, args: dict[str, Any]) -> Any:
    """Runs one tool call through a real Agent with a scripted model, returning the tool's return value."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool, args)])
        return ModelResponse(parts=[TextPart("done")])

    result = Agent(FunctionModel(model), toolsets=[toolset]).run_sync("go")
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    return returns[0].content


def test_tool_schemas() -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = {t.name: t for t in info.function_tools}
        assert set(tools) == {"contextfetch_search", "contextfetch_read"}
        search = tools["contextfetch_search"]
        assert search.parameters_json_schema["required"] == ["query"]
        assert "YYYY-MM-DD" in search.parameters_json_schema["properties"]["start_published_date"]["description"]
        assert tools["contextfetch_read"].parameters_json_schema["required"] == ["url"]
        return ModelResponse(parts=[TextPart("done")])

    Agent(FunctionModel(model), toolsets=[ContextFetchToolset("k")]).run_sync("go")


def test_search(calls: list[dict[str, Any]]) -> None:
    out = _call(ContextFetchToolset("k", num_results=2), "contextfetch_search", {"query": "q", "include_domains": ["example.com"]})
    assert out == [
        {"title": "A", "url": "https://example.com/a", "published_date": "2026-01-02", "content": "chunk A"},
        {"title": None, "url": "https://example.com/b", "published_date": None, "content": "chunk B"},
    ]
    assert calls == [
        {
            "method": "POST",
            "path": "/v1/search",
            "body": {"query": "q", "numResults": 2, "includeDomains": ["example.com"], "contents": {"pruned_context": True}},
            "api_key": "k",
        }
    ]


def test_search_text_content_and_env_key(calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTFETCH_API_KEY", "env-key")
    out = _call(ContextFetchToolset(num_results=1, content="text"), "contextfetch_search", {"query": "q"})
    assert out[0]["content"] == "# A"
    assert calls[0]["body"]["contents"] == {"text": True}
    assert calls[0]["api_key"] == "env-key"


def test_read(calls: list[dict[str, Any]]) -> None:
    out = _call(ContextFetchToolset("k"), "contextfetch_read", {"url": "https://example.com/x", "query": "pricing"})
    assert out == "# Page"
    assert calls[0]["path"] == "/r/https://example.com/x?q=pricing"


def test_read_bad_url_is_retried(calls: list[dict[str, Any]]) -> None:
    toolset = ContextFetchToolset("k")
    read = toolset.tools["contextfetch_read"].function
    with pytest.raises(ModelRetry, match="not found"):
        read("https://missing.example")


def test_read_auth_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(self: ContextFetch, *a: Any, **k: Any) -> Any:
        raise ContextFetchError("bad key", status=401)

    monkeypatch.setattr(ContextFetch, "_request", request)
    read = ContextFetchToolset("k").tools["contextfetch_read"].function
    with pytest.raises(ContextFetchError, match="bad key"):
        read("https://example.com")
