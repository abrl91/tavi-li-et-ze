"""Eight-stage Explorer Brief pipeline.

  1. CRAWL     — Tavily crawl on canonical sources (cached, 1-hour TTL, optional)
  2. SEARCH    — Tavily news search per watchlist topic (cached, 15-min TTL)
  3. URL DEDUP — collapse exact-URL duplicates across crawl+search (kept highest score)
  4. EMBED     — Nebius embedding model on (title + snippet) (cached, no expiry)
  5. NOVELTY   — drop items whose max-cosine vs prior corpus exceeds threshold
  6. RANK      — concurrent LLM call per topic picks top-k from survivors
  7. EXTRACT   — Tavily batch extract of chosen URLs for fuller content
  8. WRITE     — concurrent LLM call per chosen item produces the Explorer block

Returns a dict with the date, markdown, entries, a stats dict (JSON-safe), and
the live `Profiler` instance for the CLI to render.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_t
from typing import Iterable

import numpy as np
from pydantic import BaseModel

from .brief import render
from .profiler import Profiler
from .prompts import RANK_SYSTEM, RANK_USER, WRITE_SYSTEM, WRITE_USER
from .providers import (
    Nebius,
    SearchResult,
    TAVILY_CRAWL_BASIC_CREDITS_PER_PAGE,
    TAVILY_EXTRACT_BASIC_CREDITS_PER_URL,
    TAVILY_SEARCH_ADVANCED_CREDITS,
    Tavily,
    Usage,
)
from .store import Store


_log = logging.getLogger(__name__)
_SEARCH_CACHE_TTL_SECONDS = 15 * 60
_CRAWL_CACHE_TTL_SECONDS = 60 * 60
_DEFAULT_MAX_CONCURRENCY = 5


class RankPick(BaseModel):
    """One entry of the ranker's output: a chosen URL and one-line reason."""

    url: str
    reason: str


class RankResponse(BaseModel):
    """Strict response shape for the ranker, enforced via OpenAI SDK parse()."""

    picks: list[RankPick]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def _cosine_max_normalized(query: np.ndarray, corpus_normalized: np.ndarray) -> float:
    q = query / (np.linalg.norm(query) + 1e-9)
    return float((corpus_normalized @ q).max())


def _embed_text(item: SearchResult) -> str:
    body = item.content or item.raw_content or ""
    return f"{item.title}\n\n{body[:1500]}"


def _dedup_by_url(items: list[SearchResult]) -> list[SearchResult]:
    """Collapse same-URL duplicates within one run, keeping the highest score.

    Insertion order of first-seen URLs is preserved so downstream stages get a
    stable iteration order across runs.
    """
    by_url: dict[str, SearchResult] = {}
    for item in items:
        prev = by_url.get(item.url)
        if prev is None or item.score > prev.score:
            by_url[item.url] = item
    return list(by_url.values())


def _cached_search(
    tavily: Tavily,
    store: Store,
    profiler: Profiler,
    *,
    topic_label: str,
    query: str,
    max_results: int,
    time_range: str = "week",
    exclude_domains: list[str] | None = None,
) -> list[SearchResult]:
    """Tavily search with a 15-min SQLite cache keyed on the query params.

    `topic_label` is the watchlist label kept on each result for downstream
    grouping. `query` is what Tavily actually sees, mapped via
    `Settings.topic_queries` so bare labels like "RAG eval" expand to a richer
    query string.
    """
    exclude_key = ",".join(sorted(exclude_domains)) if exclude_domains else ""
    key = "search:" + hashlib.sha256(
        f"{query}\x00{topic_label}\x00{max_results}\x00{time_range}\x00{exclude_key}".encode("utf-8")
    ).hexdigest()
    cached = store.cache_get(key)
    if cached is not None:
        profiler.cache_event("search", hit=True)
        data = json.loads(cached.decode("utf-8"))
        return [SearchResult.model_validate(d) for d in data]
    profiler.cache_event("search", hit=False)
    results = tavily.search(
        query,
        topic_label=topic_label,
        max_results=max_results,
        time_range=time_range,
        exclude_domains=exclude_domains,
    )
    profiler.add_tavily_credits(TAVILY_SEARCH_ADVANCED_CREDITS)
    store.cache_set(
        key,
        json.dumps([r.model_dump() for r in results]).encode("utf-8"),
        ttl_seconds=_SEARCH_CACHE_TTL_SECONDS,
    )
    return results


def _cached_crawl(
    tavily: Tavily,
    store: Store,
    profiler: Profiler,
    *,
    root_url: str,
    instructions: str,
    max_depth: int = 1,
    limit: int = 10,
) -> list[SearchResult]:
    """Tavily crawl with a 1-hour SQLite cache keyed on crawl params.

    Caches longer than search because canonical-source indexes change slowly
    (hours, not minutes) and crawl is the most expensive Tavily call.
    """
    key = "crawl:" + hashlib.sha256(
        f"{root_url}\x00{instructions}\x00{max_depth}\x00{limit}".encode("utf-8")
    ).hexdigest()
    cached = store.cache_get(key)
    if cached is not None:
        profiler.cache_event("crawl", hit=True)
        data = json.loads(cached.decode("utf-8"))
        return [SearchResult.model_validate(d) for d in data]
    profiler.cache_event("crawl", hit=False)
    results = tavily.crawl(
        root_url,
        instructions=instructions,
        max_depth=max_depth,
        limit=limit,
    )
    profiler.add_tavily_credits(TAVILY_CRAWL_BASIC_CREDITS_PER_PAGE * len(results))
    store.cache_set(
        key,
        json.dumps([r.model_dump() for r in results]).encode("utf-8"),
        ttl_seconds=_CRAWL_CACHE_TTL_SECONDS,
    )
    return results


def _cached_embed(
    nebius: Nebius,
    store: Store,
    profiler: Profiler,
    texts: list[str],
) -> tuple[np.ndarray, Usage | None]:
    """Per-text SQLite cache (no expiry); only un-cached texts hit the API."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), None

    cached_vecs: list[np.ndarray | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text in enumerate(texts):
        key = _embed_cache_key(nebius.embed_model, text)
        cached = store.cache_get(key)
        if cached is not None:
            profiler.cache_event("embed", hit=True)
            cached_vecs[i] = np.frombuffer(cached, dtype=np.float32)
        else:
            profiler.cache_event("embed", hit=False)
            miss_indices.append(i)
            miss_texts.append(text)

    usage: Usage | None = None
    if miss_texts:
        fresh, usage = nebius.embed(miss_texts)
        for j, i in enumerate(miss_indices):
            vec = fresh[j].astype(np.float32)
            cached_vecs[i] = vec
            store.cache_set(
                _embed_cache_key(nebius.embed_model, miss_texts[j]),
                vec.tobytes(),
                ttl_seconds=None,
            )

    return np.stack(cached_vecs), usage


def _embed_cache_key(model: str, text: str) -> str:
    return "embed:" + hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def _rank_one_topic(
    nebius: Nebius,
    topic: str,
    items: list[SearchResult],
    k: int,
    *,
    profiler: Profiler,
    model: str | None = None,
    debug: bool = False,
) -> list[tuple[SearchResult, str]]:
    """Rank one topic's items via strict-schema Pydantic structured output.

    Returns (item, reason) pairs in the order the model picked them, capped at
    `k`. Picks whose URL isn't in `items` are dropped, defending against the
    model fabricating URLs.
    """
    if not items:
        return []
    k = min(k, len(items))
    payload = [
        {"url": it.url, "title": it.title, "snippet": (it.content or "")[:300]}
        for it in items
    ]
    user_message = RANK_USER.format(
        topic=topic, k=k, items=json.dumps(payload, indent=2)
    )
    messages = [
        {"role": "system", "content": RANK_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    if debug:
        print(f"\n{'=' * 70}\nRANK - topic={topic!r}, k={k}, n_items={len(items)}\n{'=' * 70}")
        print("SYSTEM:\n" + RANK_SYSTEM)
        print("\nUSER:\n" + user_message)

    parsed, usage = nebius.chat_parsed(
        messages,
        RankResponse,
        model=model,
        temperature=0.2,
        max_tokens=600,
    )
    profiler.add_usage(usage)

    if debug:
        print(f"\n--- RANK RESPONSE - topic={topic!r} ---")
        print(parsed.model_dump_json(indent=2) if parsed is not None else "<refusal>")

    if parsed is None:
        _log.warning(
            "RANK refused: model=%s topic=%r returned no parsed content",
            usage.model, topic,
        )
        return []

    by_url = {it.url: it for it in items}
    chosen: list[tuple[SearchResult, str]] = []
    for pick in parsed.picks[:k]:
        if pick.url in by_url:
            chosen.append((by_url[pick.url], pick.reason.strip()))
    return chosen


def _write_one_entry(
    nebius: Nebius,
    item: SearchResult,
    *,
    profiler: Profiler,
    model: str | None = None,
    extracted_content: str | None = None,
    debug: bool = False,
) -> str:
    """Write one Explorer-format entry for one chosen item."""
    body = extracted_content or item.raw_content or item.content or ""
    content = body[:3000]
    user_message = WRITE_USER.format(title=item.title, url=item.url, content=content)
    messages = [
        {"role": "system", "content": WRITE_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    if debug:
        print(f"\n{'=' * 70}\nWRITE - url={item.url}\n{'=' * 70}")
        print("SYSTEM:\n" + WRITE_SYSTEM)
        print("\nUSER:\n" + user_message)

    text, usage = nebius.chat(
        messages,
        model=model,
        temperature=0.4,
        max_tokens=400,
    )
    profiler.add_usage(usage)

    if debug:
        print(f"\n--- WRITE RESPONSE - url={item.url} ---")
        print(text)

    return text.strip()


def run(
    topics: Iterable[str],
    *,
    tavily: Tavily,
    nebius: Nebius,
    store: Store | None = None,
    dedup_threshold: float = 0.86,
    max_results_per_topic: int = 10,
    top_k_per_topic: int = 3,
    rank_model: str | None = None,
    write_model: str | None = None,
    topic_queries: dict[str, str] | None = None,
    canonical_sources: list[dict[str, str]] | None = None,
    search_exclude_domains: list[str] | None = None,
    enable_crawl: bool = True,
    search_time_range: str = "week",
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    debug: bool = False,
) -> dict:
    store = store or Store()
    profiler = Profiler()
    topic_queries = topic_queries or {}
    canonical_sources = canonical_sources or []

    topics = list(topics)
    today = date_t.today().isoformat()

    # Crawl runs first because it has the longest timeout and is the most
    # failure-prone Tavily endpoint, so a hang here doesn't strand the pipeline
    # mid-flight.
    raw: list[SearchResult] = []
    if enable_crawl and canonical_sources:
        for source in canonical_sources:
            with profiler.stage("crawl"):
                raw.extend(
                    _cached_crawl(
                        tavily, store, profiler,
                        root_url=source["url"],
                        instructions=source.get("instructions", ""),
                    )
                )

    for topic in topics:
        query = topic_queries.get(topic, topic)
        with profiler.stage("search"):
            raw.extend(
                _cached_search(
                    tavily, store, profiler,
                    topic_label=topic,
                    query=query,
                    max_results=max_results_per_topic,
                    time_range=search_time_range,
                    exclude_domains=search_exclude_domains,
                )
            )

    candidates = _dedup_by_url(raw)

    with profiler.stage("embed"):
        embeddings, embed_usage = _cached_embed(
            nebius, store, profiler, [_embed_text(c) for c in candidates]
        )
        if embed_usage is not None:
            profiler.add_usage(embed_usage)

    with profiler.stage("novelty"):
        prior_mat, _ = store.all_embeddings()
        prior_normalized = (
            _normalize(prior_mat) if prior_mat is not None and prior_mat.size else None
        )
        survivors: list[tuple[SearchResult, np.ndarray]] = []
        n_dropped_novelty = 0
        for cand, emb in zip(candidates, embeddings):
            if (
                prior_normalized is not None
                and prior_normalized.shape[1] == emb.shape[0]
                and _cosine_max_normalized(emb, prior_normalized) > dedup_threshold
            ):
                n_dropped_novelty += 1
                continue
            survivors.append((cand, emb))

    by_topic: dict[str, list[SearchResult]] = {}
    for it, _ in survivors:
        by_topic.setdefault(it.topic, []).append(it)

    chosen: list[tuple[SearchResult, str]] = []
    if by_topic:
        with profiler.stage("rank"):
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                futures = [
                    ex.submit(
                        _rank_one_topic,
                        nebius, topic, items, top_k_per_topic,
                        profiler=profiler, model=rank_model, debug=debug,
                    )
                    for topic, items in by_topic.items()
                ]
                for f in futures:
                    try:
                        chosen.extend(f.result())
                    except Exception:
                        _log.exception("rank task failed, skipping")

    chosen_urls = [it.url for it, _ in chosen]
    extracted: dict[str, str] = {}
    if chosen_urls:
        with profiler.stage("extract"):
            extracted = tavily.extract(chosen_urls)
            profiler.add_tavily_credits(
                TAVILY_EXTRACT_BASIC_CREDITS_PER_URL * len(extracted)
            )

    entries: list[dict] = []
    if chosen:
        with profiler.stage("write"):
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                futures = [
                    (
                        it, reason,
                        ex.submit(
                            _write_one_entry,
                            nebius, it,
                            profiler=profiler,
                            model=write_model,
                            extracted_content=extracted.get(it.url),
                            debug=debug,
                        ),
                    )
                    for it, reason in chosen
                ]
                for it, reason, f in futures:
                    try:
                        md = f.result()
                    except Exception:
                        _log.exception("write task failed for %s, skipping", it.url)
                        continue
                    entries.append(
                        {
                            "url": it.url,
                            "title": it.title,
                            "topic": it.topic,
                            "reason": reason,
                            "markdown": md,
                        }
                    )

    if survivors:
        store.save_items(
            [s[0] for s in survivors],
            np.stack([s[1] for s in survivors]),
            first_seen=today,
        )

    markdown = render(
        entries,
        today,
        n_searched=len(raw),
        n_dropped_novelty=n_dropped_novelty,
    )
    stats = {
        "date": today,
        "topics": topics,
        "n_searched": len(raw),
        "n_after_url_dedup": len(candidates),
        "n_after_novelty": len(survivors),
        "n_dropped_novelty": n_dropped_novelty,
        "n_in_brief": len(entries),
        "n_extracted": len(extracted),
        "dedup_threshold": dedup_threshold,
        "profiler": profiler.to_dict(),
    }
    store.save_brief(today, markdown, stats)
    return {
        "date": today,
        "markdown": markdown,
        "entries": entries,
        "stats": stats,
        "profiler": profiler,
    }
