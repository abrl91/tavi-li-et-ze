"""Tavily (search) and Nebius Token Factory (LLM + embeddings) clients.

Both take explicit values in __init__. Env reads happen in the CLI via Settings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Literal

import httpx
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient as _TavilyClient
from tavily.errors import (
    TimeoutError as _TavilyTimeoutError,
    UsageLimitExceededError as _TavilyUsageLimitError,
)


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

# Per-call Tavily timeouts (seconds). Crawl is the slowest, then extract, then
# search. These are passed to the Tavily SDK per call.
_TAVILY_SEARCH_TIMEOUT_S: int = 30
_TAVILY_EXTRACT_TIMEOUT_S: int = 60
_TAVILY_CRAWL_TIMEOUT_S: int = 90

# Retry policy for Tavily transient failures (rate limit, timeout). Three
# attempts with 1s, 2s, 4s backoff covers most spurious failures without
# blocking the pipeline for long.
_TAVILY_RETRY_ATTEMPTS: int = 3
_TAVILY_RETRY_BASE_DELAY_S: float = 1.0

# Defensive cap on embedding batch size. Nebius does not document a hard
# per-request limit but very large batches risk 413 or rate-limit errors.
_EMBED_BATCH_SIZE: int = 100

# Nebius HTTP client timeouts. Read is the dominant cost on LLM calls.
_NEBIUS_TOTAL_TIMEOUT_S: float = 60.0
_NEBIUS_READ_TIMEOUT_S: float = 45.0
_NEBIUS_WRITE_TIMEOUT_S: float = 10.0
_NEBIUS_CONNECT_TIMEOUT_S: float = 5.0
_NEBIUS_MAX_RETRIES: int = 3


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


def _tavily_with_retry(call):
    """Retry a Tavily call up to N times with exponential backoff.

    Only retries on transient errors (rate limit, timeout). Other errors
    propagate immediately so we don't burn money or time on permanent failures.
    """
    last_err: Exception | None = None
    for attempt in range(_TAVILY_RETRY_ATTEMPTS):
        try:
            return call()
        except (_TavilyUsageLimitError, _TavilyTimeoutError) as err:
            last_err = err
            if attempt < _TAVILY_RETRY_ATTEMPTS - 1:
                time.sleep(_TAVILY_RETRY_BASE_DELAY_S * (2 ** attempt))
    assert last_err is not None
    raise last_err


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
            "timeout": _TAVILY_SEARCH_TIMEOUT_S,
        }
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        response = _tavily_with_retry(lambda: self._client.search(**kwargs))
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
            response = _tavily_with_retry(
                lambda b=batch: self._client.extract(
                    urls=b,
                    extract_depth=extract_depth,
                    format="markdown",
                    timeout=_TAVILY_EXTRACT_TIMEOUT_S,
                )
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
        response = _tavily_with_retry(
            lambda: self._client.crawl(
                url=root_url,
                instructions=instructions,
                max_depth=max_depth,
                limit=limit,
                extract_depth="basic",
                format="markdown",
                chunks_per_source=3,
                timeout=_TAVILY_CRAWL_TIMEOUT_S,
            )
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

        Explicit timeout and retry settings replace the SDK defaults (10-minute
        total timeout, 2 retries) which are too forgiving for a background
        pipeline. Retries fire on connection errors, 408, 409, 429, and >=500.

        Args:
            api_key: Nebius API key.
            llm_model: Default chat-completion model (e.g. `meta-llama/Llama-3.3-70B-Instruct`).
            embed_model: Default embedding model (e.g. `Qwen/Qwen3-Embedding-8B`).
            base_url: Token Factory base URL; trailing slash is stripped.
        """
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(
                _NEBIUS_TOTAL_TIMEOUT_S,
                read=_NEBIUS_READ_TIMEOUT_S,
                write=_NEBIUS_WRITE_TIMEOUT_S,
                connect=_NEBIUS_CONNECT_TIMEOUT_S,
            ),
            max_retries=_NEBIUS_MAX_RETRIES,
        )
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
                Use `system` for role/rules/style, `user` for task/data.
            model: Per-call override. Defaults to `self.llm_model` if None.
                Lets callers route rank vs write to different models.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Cap on response length.
            response_format: Optional structured-output spec. For strict-schema
                output use `{"type": "json_schema", "json_schema": {"name": ...,
                "strict": True, "schema": {...}}}`; for unstructured JSON use
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

    def chat_parsed(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> tuple[BaseModel | None, Usage]:
        """Strict-schema chat. Returns a Pydantic instance parsed from the response.

        Uses the OpenAI SDK's `chat.completions.parse()` which auto-converts the
        Pydantic model to a strict JSON schema (additionalProperties=false, all
        fields required) and sends it as `response_format`. The Nebius server
        sees the same `{"type": "json_schema", ...}` shape its docs document.

        Args:
            messages: System+user message list.
            response_model: Pydantic model class describing the response shape.
            model: Per-call model override.
            temperature: Sampling temperature.
            max_tokens: Cap on response length.

        Returns:
            Tuple of `(parsed_instance, Usage)`. `parsed_instance` is `None` if
            the model refused (safety filter) — caller decides how to handle.
        """
        chosen_model = model or self.llm_model
        completion = self._client.chat.completions.parse(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_model,
        )
        usage = Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            model=chosen_model,
        )
        return completion.choices[0].message.parsed, usage

    def embed(self, texts: Iterable[str]) -> tuple[np.ndarray, Usage]:
        """Embed a batch of texts in one or more API calls.

        Inputs are chunked at `_EMBED_BATCH_SIZE` per request so large batches
        don't risk 413 Request Too Large or hit undocumented per-request caps.

        Args:
            texts: Strings to embed (consumed once if it's a generator).

        Returns:
            Tuple of `(matrix, Usage)`. `matrix` is `np.ndarray` of shape
            `(len(texts), embed_dim)` and dtype float32. Usage aggregates
            prompt tokens across all chunked calls.
        """
        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32), Usage(
                prompt_tokens=0, completion_tokens=0, model=self.embed_model
            )

        all_vectors: list[list[float]] = []
        total_prompt_tokens = 0
        for start in range(0, len(items), _EMBED_BATCH_SIZE):
            batch = items[start:start + _EMBED_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.embed_model, input=batch
            )
            all_vectors.extend(item.embedding for item in response.data)
            total_prompt_tokens += response.usage.prompt_tokens

        embeddings = np.asarray(all_vectors, dtype=np.float32)
        usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            model=self.embed_model,
        )
        return embeddings, usage
