"""URL dedup: within one run, collapse same-URL hits and keep the best-scored copy."""

from __future__ import annotations

from ai_news_scout.pipeline import _dedup_by_url


def test_empty_input_is_a_noop():
    assert _dedup_by_url([]) == []


def test_distinct_urls_are_all_preserved(make_result):
    items = [
        make_result("https://a.test/1", score=0.5),
        make_result("https://b.test/2", score=0.7),
        make_result("https://c.test/3", score=0.3),
    ]
    out = _dedup_by_url(items)
    assert {r.url for r in out} == {
        "https://a.test/1",
        "https://b.test/2",
        "https://c.test/3",
    }


def test_duplicate_url_keeps_highest_score_copy(make_result):
    same_url = "https://shared.test/post"
    items = [
        make_result(same_url, topic="t1", score=0.4, title="t1 copy"),
        make_result(same_url, topic="t2", score=0.9, title="t2 copy"),
        make_result(same_url, topic="t3", score=0.6, title="t3 copy"),
    ]
    out = _dedup_by_url(items)
    assert len(out) == 1
    winner = out[0]
    assert winner.score == 0.9
    assert winner.title == "t2 copy"
    assert winner.topic == "t2"


def test_first_seen_kept_when_scores_tie(make_result):
    same_url = "https://shared.test/post"
    items = [
        make_result(same_url, topic="t1", score=0.5, title="first"),
        make_result(same_url, topic="t2", score=0.5, title="second"),
    ]
    out = _dedup_by_url(items)
    assert len(out) == 1
    # Strict ">" tiebreak means the first copy wins.
    assert out[0].title == "first"
