"""ContextFetch web search and page reading tools for Pydantic AI."""

from __future__ import annotations

from typing import Any, Literal

from contextfetch import ContextFetch, ContextFetchError
from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

__all__ = ("ContextFetchToolset",)

Content = Literal["pruned_context", "text"]
"""`"pruned_context"`: only the parts of each page relevant to the query. `"text"`: the whole page as Markdown."""

_RESULT_KEY = {"pruned_context": "prunedContext", "text": "text"}


class ContextFetchToolset(FunctionToolset[Any]):
    """A toolset with two tools that share one ContextFetch client.

    - `contextfetch_search`: web search; each result carries the page content as Markdown.
    - `contextfetch_read`: one URL as Markdown, optionally only the parts relevant to a question.

    ```python
    from pydantic_ai import Agent

    from pydantic_ai_contextfetch import ContextFetchToolset

    agent = Agent('openai:gpt-5.2', toolsets=[ContextFetchToolset()])
    ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        num_results: int = 5,
        content: Content = "pruned_context",
        id: str | None = None,
    ):
        """Creates the toolset.

        Args:
            api_key: The ContextFetch API key. Defaults to the `CONTEXTFETCH_API_KEY` environment variable;
                without either, requests use the keyless tier.
            num_results: Results per search (1 to 25). Defaults to 5.
            content: Which page content search results carry. Defaults to `"pruned_context"`.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        client = ContextFetch(api_key)
        key = _RESULT_KEY[content]

        def contextfetch_search(
            query: str,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
            start_published_date: str | None = None,
        ) -> list[dict[str, Any]]:
            """Searches the web and returns the top results with their page content as Markdown.

            Args:
                query: The web search query.
                include_domains: Only return results from these domains.
                exclude_domains: Never return results from these domains.
                start_published_date: Only return pages published on or after this date (YYYY-MM-DD).
            """
            response = client.search(
                query,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_published_date=start_published_date,
                text=content == "text",
                pruned_context=content == "pruned_context",
            )
            return [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "published_date": r.get("publishedDate"),
                    "content": r.get(key) or "",
                }
                for r in response.get("results", [])
            ]

        def contextfetch_read(url: str, query: str | None = None) -> str:
            """Reads a web page and returns it as Markdown.

            Args:
                url: The URL of the page.
                query: If set, only the parts of the page relevant to this question are returned.
            """
            try:
                return client.read(url, query=query).markdown
            except ContextFetchError as e:
                # A bad or unreachable URL is something the model can fix; auth, credit and server errors are not.
                if e.status in (400, 404, 422):
                    raise ModelRetry(str(e)) from e
                raise

        super().__init__([contextfetch_search, contextfetch_read], id=id)
