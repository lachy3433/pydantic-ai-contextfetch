# pydantic-ai-contextfetch

[ContextFetch](https://contextfetch.dev) web search and page reading tools for [Pydantic AI](https://ai.pydantic.dev), as a toolset.

```bash
pip install pydantic-ai-contextfetch
export CONTEXTFETCH_API_KEY="your-api-key"  # optional: without a key, requests use the keyless tier
```

```python
from pydantic_ai import Agent

from pydantic_ai_contextfetch import ContextFetchToolset

agent = Agent('openai:gpt-5.2', toolsets=[ContextFetchToolset()])
result = agent.run_sync('What changed in the latest Python release?')
print(result.output)
```

## Tools

- `contextfetch_search(query, include_domains?, exclude_domains?, start_published_date?)`: web search. Returns a list
  of `{title, url, published_date, content}`, where `content` is the page as Markdown: by default only the parts
  relevant to the query.
- `contextfetch_read(url, query?)`: one page as Markdown. With `query`, only the parts relevant to it. A bad or
  unreachable URL raises `ModelRetry`, so the model can correct it.

## Options

```python
ContextFetchToolset(
    api_key=None,              # defaults to CONTEXTFETCH_API_KEY
    num_results=5,             # results per search, 1 to 25
    content='pruned_context',  # or 'text' for whole pages
    id=None,                   # toolset id for durable execution
)
```

To expose only one tool, use Pydantic AI's toolset filtering:
`ContextFetchToolset().filtered(lambda ctx, tool: tool.name == 'contextfetch_search')`.

## Tests

```bash
uv run --group test pytest tests                                   # unit tests, no network
CONTEXTFETCH_API_KEY=... uv run --group test pytest tests/test_live.py  # live, 2 credits
```

MIT licensed.
