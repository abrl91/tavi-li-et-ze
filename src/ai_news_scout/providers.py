"""Tavily (search) and Nebius Token Factory (LLM + embeddings) clients.

Both take explicit values in __init__ — env reads happen in the CLI via Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient as _TavilyClient


PRICE_PER_MILLION_TOKENS_USD: dict[str, dict[str, float]] = {
    "meta-llama/Llama-3.3-70B-Instruct":     {"in": 0.13,  "out": 0.40},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"in": 0.02,  "out": 0.06},
    "Qwen/Qwen3-30B-A3B-Instruct-2507":      {"in": 0.10,  "out": 0.30},
    "Qwen/Qwen3-Embedding-8B":               {"in": 0.025, "out": 0.0},
}


# Tavily Pay-As-You-Go pricing: roughly $0.008 per API credit.
# Credit costs per docs (https://docs.tavily.com/documentation/api-credits):
#   - /search advanced            : 2 credits/request
#   - /extract basic              : 0.2 credits/successful URL (1 per 5)
#   - /crawl basic                : 0.4 credits/extracted page (0.2 map + 0.2 extract)
# Used by Profiler to estimate dollar cost of the Tavily side of the pipeline.
TAVILY_CREDIT_USD: float = 0.008
TAVILY_SEARCH_ADVANCED_CREDITS: float = 2.0
TAVILY_EXTRACT_BASIC_CREDITS_PER_URL: float = 0.2
TAVILY_CRAWL_BASIC_CREDITS_PER_PAGE: float = 0.4

# Tavily /extract endpoint caps each request at 20 URLs (returns 400 above).
# Higher than this and Tavily.extract will chunk internally.
_TAVILY_EXTRACT_MAX_URLS_PER_CALL: int = 20


TavilyTopic = Literal["general", "news", "finance"]


class SearchResult(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0
    raw_content: str | None = None
    topic: str
    source: Literal["search", "crawl"] = "search"


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    model: str


class Tavily:
    def __init__(self, api_key: str):
        """Wrap the Tavily SDK with our defaults.

        Args:
            api_key: Tavily API key.
        """
        self._client = _TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        *,
        topic_label: str,
        topic: TavilyTopic = "news",
        time_range: str = "week",
        max_results: int = 10,
        exclude_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        """Run an advanced Tavily search and return validated results.

        Args:
            query: Search query string sent to Tavily.
            topic_label: Our watchlist label, attached to each result for
                downstream grouping (independent of Tavily's `topic`).
            topic: Tavily's content bias (`news`, `general`, or `finance`).
            time_range: Recency window (`day`, `week`, `month`, `year`).
            max_results: Cap on hits returned.
            exclude_domains: Domains to filter out, e.g.
                `["medium.com", "dev.to"]`. None means no filter.

        Returns:
            List of `SearchResult`, possibly empty.
        """
        kwargs: dict = {
            "query": query,
            "search_depth": "advanced",
            "topic": topic,
            "time_range": time_range,
            "max_results": max_results,
            "include_raw_content": "markdown",
        }
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        response = self._client.search(**kwargs)
        return [
            SearchResult.model_validate(
                {**result, "topic": topic_label, "source": "search"}
            )
            for result in response.get("results", [])
        ]

    def extract(
        self,
        urls: list[str],
        *,
        extract_depth: Literal["basic", "advanced"] = "basic",
    ) -> dict[str, str]:
        """Batch-extract clean Markdown bodies for one or more URLs.

        Tavily caps each /extract call at 20 URLs. This wrapper chunks larger
        inputs into multiple calls and stitches the results back together.

        Args:
            urls: List of URLs to extract content from. Order does not matter,
                results are returned keyed by URL.
            extract_depth: Tavily extract depth. `basic` is ~$0.0016/URL,
                `advanced` doubles cost for a fuller crawl.

        Returns:
            Dict mapping URL to extracted Markdown body. URLs that Tavily
            failed to extract are omitted from the dict (no exception).
        """
        out: dict[str, str] = {}
        for start in range(0, len(urls), _TAVILY_EXTRACT_MAX_URLS_PER_CALL):
            batch = urls[start:start + _TAVILY_EXTRACT_MAX_URLS_PER_CALL]
            response = self._client.extract(
                urls=batch,
                extract_depth=extract_depth,
                format="markdown",
            )
            for result in response.get("results", []):
                url = result.get("url")
                body = result.get("raw_content")
                if url and body:
                    out[url] = body
        return out

    def crawl(
        self,
        root_url: str,
        *,
        instructions: str,
        max_depth: int = 1,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Crawl `root_url` and return up to `limit` pages as `SearchResult`s.

        Args:
            root_url: Domain entry point (e.g. `https://www.anthropic.com/news`).
            instructions: Natural-language guidance for the crawler about what
                to keep (e.g. "Recent posts about model releases").
            max_depth: Link-following depth. `1` means root + one hop.
            limit: Max number of pages to extract.

        Returns:
            List of `SearchResult` with `source="crawl"` and `topic` set to
            `root_url` (used by downstream stages for grouping and attribution).
            Empty list if the crawl returned no pages.
        """
        response = self._client.crawl(
            url=root_url,
            instructions=instructions,
            max_depth=max_depth,
            limit=limit,
            extract_depth="basic",
            format="markdown",
        )
        results: list[SearchResult] = []
        for page in response.get("results", []):
            url = page.get("url")
            raw = page.get("raw_content") or ""
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=url,
                    content=raw[:1500],
                    score=0.0,
                    raw_content=raw,
                    topic=root_url,
                    source="crawl",
                )
            )
        return results


class Nebius:
    def __init__(
        self,
        *,
        api_key: str,
        llm_model: str,
        embed_model: str,
        base_url: str,
    ):
        """Wrap the OpenAI SDK pointed at Nebius Token Factory.

        Args:
            api_key: Nebius API key.
            llm_model: Default chat-completion model (e.g. `meta-llama/Llama-3.3-70B-Instruct`).
            embed_model: Default embedding model (e.g. `Qwen/Qwen3-Embedding-8B`).
            base_url: Token Factory base URL; trailing slash is stripped.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        self.llm_model = llm_model
        self.embed_model = embed_model

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> tuple[str, Usage]:
        """Send a chat completion to the configured LLM.

        Args:
            messages: OpenAI-style `[{"role": ..., "content": ...}]` list.
            model: Per-call override. Defaults to `self.llm_model` if None.
                Lets callers route rank vs write to different models.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Cap on response length.
            response_format: Optional structured-output spec, e.g.
                `{"type": "json_object"}`.

        Returns:
            Tuple of `(assistant_text, Usage)` with token counts and model id.
        """
        chosen_model = model or self.llm_model
        kwargs: dict = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=chosen_model,
        )
        return response.choices[0].message.content or "", usage

    def embed(self, texts: Iterable[str]) -> tuple[np.ndarray, Usage]:
        """Embed a batch of texts in one API call.

        Args:
            texts: Strings to embed (consumed once if it's a generator).

        Returns:
            Tuple of `(matrix, Usage)`. `matrix` is `np.ndarray` of shape
            `(len(texts), embed_dim)` and dtype float32.
        """
        items = list(texts)
        response = self._client.embeddings.create(
            model=self.embed_model, input=items)
        embeddings = np.asarray(
            [item.embedding for item in response.data], dtype=np.float32
        )
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=0,
            model=self.embed_model,
        )
        return embeddings, usage
