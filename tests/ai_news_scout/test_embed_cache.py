"""Embed cache: per-text SQLite cache keyed on (model, text)."""

from __future__ import annotations

import numpy as np

from ai_news_scout.pipeline import _cached_embed, _embed_cache_key
from ai_news_scout.profiler import Profiler


def _seed_cache(store, nebius, text: str, vec: np.ndarray) -> None:
    store.cache_set(
        _embed_cache_key(nebius.embed_model, text),
        vec.astype(np.float32).tobytes(),
        ttl_seconds=None,
    )


def test_all_cached_hits_zero_api_calls(temp_store, fake_nebius):
    texts = ["alpha", "beta"]
    for t in texts:
        _seed_cache(
            temp_store,
            fake_nebius,
            t,
            np.full(fake_nebius.embed_dim, 0.1, dtype=np.float32),
        )

    profiler = Profiler()
    with profiler.stage("embed"):
        out, usage = _cached_embed(fake_nebius, temp_store, profiler, texts)

    assert fake_nebius.embed_calls == []
    assert usage is None
    assert out.shape == (2, fake_nebius.embed_dim)


def test_mixed_cache_only_misses_hit_api(temp_store, fake_nebius):
    cached_text = "already cached"
    fresh_text = "needs embedding"
    cached_vec = np.full(fake_nebius.embed_dim, 0.2, dtype=np.float32)
    _seed_cache(temp_store, fake_nebius, cached_text, cached_vec)

    profiler = Profiler()
    with profiler.stage("embed"):
        out, usage = _cached_embed(
            fake_nebius, temp_store, profiler, [cached_text, fresh_text]
        )

    assert len(fake_nebius.embed_calls) == 1
    assert fake_nebius.embed_calls[0] == [fresh_text]
    assert usage is not None
    assert np.allclose(out[0], cached_vec)
    assert not np.allclose(out[1], cached_vec)


def test_empty_input_returns_empty_matrix(temp_store, fake_nebius):
    profiler = Profiler()
    with profiler.stage("embed"):
        out, usage = _cached_embed(fake_nebius, temp_store, profiler, [])
    assert out.shape == (0, 0)
    assert usage is None
    assert fake_nebius.embed_calls == []


def test_second_call_uses_freshly_populated_cache(temp_store, fake_nebius):
    texts = ["one", "two"]
    profiler = Profiler()
    with profiler.stage("embed"):
        first, _ = _cached_embed(fake_nebius, temp_store, profiler, texts)
    # Reset call log to isolate the second invocation.
    fake_nebius.embed_calls.clear()
    with profiler.stage("embed"):
        second, _ = _cached_embed(fake_nebius, temp_store, profiler, texts)
    assert fake_nebius.embed_calls == []
    assert np.allclose(first, second)
