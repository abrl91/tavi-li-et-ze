"""End-to-end pipeline.run with fakes for every provider.

Exercises the full eight-stage flow (minus crawl, which is opt-out) and proves
the DI design holds. Run twice with identical inputs: the second pass must hit
caches and have everything dropped by novelty.
"""

from __future__ import annotations

from ai_news_scout.pipeline import run as run_pipeline


def _topic_results(make_result):
    return [
        make_result(
            "https://x.test/article-1",
            topic="t1",
            title="Article one",
            score=0.9,
        ),
        make_result(
            "https://x.test/article-2",
            topic="t1",
            title="Article two",
            score=0.7,
        ),
        make_result(
            "https://x.test/article-3",
            topic="t1",
            title="Article three",
            score=0.5,
        ),
    ]


def test_pipeline_runs_end_to_end(
    temp_store, fake_tavily, fake_nebius, make_result
):
    fake_tavily.search_results = {"t1": _topic_results(make_result)}
    fake_tavily.extract_results = {
        "https://x.test/article-1": "Full body 1",
        "https://x.test/article-2": "Full body 2",
        "https://x.test/article-3": "Full body 3",
    }

    result = run_pipeline(
        topics=["t1"],
        tavily=fake_tavily,
        nebius=fake_nebius,
        store=temp_store,
        max_results_per_topic=3,
        top_k_per_topic=2,
        enable_crawl=False,
    )

    stats = result["stats"]
    assert stats["n_searched"] == 3
    assert stats["n_after_url_dedup"] == 3
    assert stats["n_after_novelty"] == 3
    assert stats["n_in_brief"] == 2
    assert stats["n_extracted"] == 2

    stage_keys = list(stats["profiler"]["stages"].keys())
    assert stage_keys == ["search", "embed", "novelty", "rank", "extract", "write"]

    assert "Explorer Brief" in result["markdown"]
    assert result["markdown"].count("Test entry") == 2

    mat, urls = temp_store.all_embeddings()
    assert mat is not None
    assert len(urls) == 3
    assert mat.shape == (3, fake_nebius.embed_dim)

    assert len(fake_tavily.search_calls) == 1
    assert len(fake_tavily.extract_calls) == 1
    assert len(fake_tavily.extract_calls[0]) == 2
    assert len(fake_nebius.embed_calls) == 1
    rank_chats = [c for c in fake_nebius.chat_calls if c["response_format"]]
    write_chats = [c for c in fake_nebius.chat_calls if not c["response_format"]]
    assert len(rank_chats) == 1
    assert len(write_chats) == 2


def test_second_run_hits_caches_and_drops_everything(
    temp_store, fake_tavily, fake_nebius, make_result
):
    fake_tavily.search_results = {"t1": _topic_results(make_result)}
    fake_tavily.extract_results = {
        r.url: f"body for {r.url}" for r in _topic_results(make_result)
    }

    run_pipeline(
        topics=["t1"],
        tavily=fake_tavily,
        nebius=fake_nebius,
        store=temp_store,
        max_results_per_topic=3,
        top_k_per_topic=2,
        enable_crawl=False,
    )

    search_calls_after_run1 = len(fake_tavily.search_calls)
    extract_calls_after_run1 = len(fake_tavily.extract_calls)
    embed_calls_after_run1 = len(fake_nebius.embed_calls)
    chat_calls_after_run1 = len(fake_nebius.chat_calls)

    result2 = run_pipeline(
        topics=["t1"],
        tavily=fake_tavily,
        nebius=fake_nebius,
        store=temp_store,
        max_results_per_topic=3,
        top_k_per_topic=2,
        enable_crawl=False,
    )

    assert len(fake_tavily.search_calls) == search_calls_after_run1
    assert len(fake_nebius.embed_calls) == embed_calls_after_run1
    assert len(fake_nebius.chat_calls) == chat_calls_after_run1
    assert len(fake_tavily.extract_calls) == extract_calls_after_run1

    stats2 = result2["stats"]
    assert stats2["n_searched"] == 3
    assert stats2["n_after_novelty"] == 0
    assert stats2["n_dropped_novelty"] == 3
    assert stats2["n_in_brief"] == 0

    stage_keys = list(stats2["profiler"]["stages"].keys())
    assert stage_keys == ["search", "embed", "novelty"]

    caches = stats2["profiler"]["caches"]
    assert caches["search"]["hits"] >= 1
    assert caches["search"]["misses"] == 0
    assert caches["embed"]["hits"] >= 1
    assert caches["embed"]["misses"] == 0


def test_url_dedup_collapses_same_url_across_topics(
    temp_store, fake_tavily, fake_nebius, make_result
):
    shared_url = "https://shared.test/same-post"
    fake_tavily.search_results = {
        "t1": [make_result(shared_url, topic="t1", score=0.4, title="t1 view")],
        "t2": [make_result(shared_url, topic="t2", score=0.9, title="t2 view")],
    }

    result = run_pipeline(
        topics=["t1", "t2"],
        tavily=fake_tavily,
        nebius=fake_nebius,
        store=temp_store,
        max_results_per_topic=5,
        top_k_per_topic=2,
        enable_crawl=False,
    )

    assert result["stats"]["n_searched"] == 2
    assert result["stats"]["n_after_url_dedup"] == 1
