"""Shared test fixtures and provider fakes.

The pipeline takes its providers via keyword args, so tests inject fakes that
implement the same surface without touching the network. FakeTavily records
every call and returns predetermined results. FakeNebius routes RANK calls
through `chat_parsed` (which returns a Pydantic instance) and WRITE calls
through `chat` (which returns plain text). Embeddings are deterministic via
SHA-256 seeding so the same text always gets the same vector across runs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pytest
from pydantic import BaseModel

from ai_news_scout.providers import SearchResult, Usage
from ai_news_scout.store import Store


_FAKE_EMBED_DIM = 64
_FAKE_LLM_MODEL = "fake-llm"
_FAKE_EMBED_MODEL = "fake-embed"


def deterministic_vector(text: str, dim: int = _FAKE_EMBED_DIM) -> np.ndarray:
    """Map any text to the same unit vector every time."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class FakeTavily:
    """In-memory Tavily double. Routes search/crawl/extract via preset tables."""

    search_results: dict[str, list[SearchResult]] = field(default_factory=dict)
    crawl_results: dict[str, list[SearchResult]] = field(default_factory=dict)
    extract_results: dict[str, str] = field(default_factory=dict)

    search_calls: list[dict] = field(default_factory=list)
    crawl_calls: list[dict] = field(default_factory=list)
    extract_calls: list[list[str]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        topic_label: str,
        topic: str = "news",
        time_range: str = "week",
        max_results: int = 10,
        exclude_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "query": query,
                "topic_label": topic_label,
                "max_results": max_results,
                "exclude_domains": exclude_domains,
            }
        )
        return list(self.search_results.get(topic_label, []))[:max_results]

    def crawl(
        self,
        root_url: str,
        *,
        instructions: str,
        max_depth: int = 1,
        limit: int = 10,
    ) -> list[SearchResult]:
        self.crawl_calls.append(
            {"root_url": root_url, "instructions": instructions, "limit": limit}
        )
        return list(self.crawl_results.get(root_url, []))[:limit]

    def extract(
        self,
        urls: list[str],
        *,
        extract_depth: Literal["basic", "advanced"] = "basic",
    ) -> dict[str, str]:
        self.extract_calls.append(list(urls))
        return {u: self.extract_results[u] for u in urls if u in self.extract_results}


@dataclass
class FakeNebius:
    """Nebius double. Chat: canned RANK/WRITE responses. Embed: hash-derived vectors."""

    llm_model: str = _FAKE_LLM_MODEL
    embed_model: str = _FAKE_EMBED_MODEL
    embed_dim: int = _FAKE_EMBED_DIM

    chat_calls: list[dict] = field(default_factory=list)
    embed_calls: list[list[str]] = field(default_factory=list)

    write_response: str = "## Test entry\n\nTest body text."

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> tuple[str, Usage]:
        chosen_model = model or self.llm_model
        prompt = messages[-1].get("content", "")
        self.chat_calls.append(
            {
                "model": chosen_model,
                "response_format": response_format,
                "prompt": prompt,
            }
        )
        usage = Usage(prompt_tokens=100, completion_tokens=50, model=chosen_model)
        return self.write_response, usage

    def chat_parsed(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> tuple[BaseModel | None, Usage]:
        """Mimic SDK structured-output parse: return a Pydantic instance built
        from URLs found in the user prompt's `Items:` block."""
        chosen_model = model or self.llm_model
        prompt = messages[-1].get("content", "")
        self.chat_calls.append(
            {
                "model": chosen_model,
                "response_format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                },
                "prompt": prompt,
            }
        )
        items_block = prompt.split("Items:", 1)[-1]
        urls = re.findall(r'"url":\s*"([^"]+)"', items_block)
        k_match = re.search(r"TOP\s+(\d+)", prompt, flags=re.IGNORECASE)
        k = int(k_match.group(1)) if k_match else len(urls)
        picks = [{"url": u, "reason": "fake pick"} for u in urls[:k]]
        parsed = response_model.model_validate({"picks": picks})
        usage = Usage(prompt_tokens=100, completion_tokens=50, model=chosen_model)
        return parsed, usage

    def embed(self, texts: Iterable[str]) -> tuple[np.ndarray, Usage]:
        items = list(texts)
        self.embed_calls.append(list(items))
        vectors = np.stack(
            [deterministic_vector(t, dim=self.embed_dim) for t in items]
        )
        usage = Usage(
            prompt_tokens=sum(len(t) for t in items),
            completion_tokens=0,
            model=self.embed_model,
        )
        return vectors, usage


@pytest.fixture
def temp_store(tmp_path) -> Store:
    """Fresh on-disk SQLite store, isolated per test."""
    return Store(tmp_path / "test.db")


@pytest.fixture
def fake_tavily() -> FakeTavily:
    return FakeTavily()


@pytest.fixture
def fake_nebius() -> FakeNebius:
    return FakeNebius()


@pytest.fixture
def make_result():
    """Returns a compact factory for `SearchResult` fixtures."""

    def _make(
        url: str,
        *,
        topic: str = "t1",
        title: str | None = None,
        content: str | None = None,
        score: float = 0.5,
        source: Literal["search", "crawl"] = "search",
    ) -> SearchResult:
        return SearchResult(
            url=url,
            title=title or f"Title for {url}",
            content=content or f"Snippet body for {url}",
            score=score,
            topic=topic,
            source=source,
        )

    return _make
