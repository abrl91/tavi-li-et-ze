"""Novelty filter primitives: normalize once, then cosine-max per candidate."""

from __future__ import annotations

import numpy as np

from ai_news_scout.pipeline import _cosine_max_normalized, _normalize


def test_orthogonal_vectors_have_near_zero_cosine():
    corpus = _normalize(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_max_normalized(query, corpus) == 0.0


def test_identical_vector_has_cosine_one():
    corpus = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_max_normalized(query, corpus) == 1.0


def test_near_duplicate_exceeds_default_threshold():
    # `0.86` is the production default. A tiny perturbation should still trigger.
    corpus = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    query = np.array([1.0, 0.05, 0.0], dtype=np.float32)
    assert _cosine_max_normalized(query, corpus) > 0.86


def test_max_picks_the_closer_neighbor():
    corpus = _normalize(
        np.array(
            [[0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            dtype=np.float32,
        )
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = _cosine_max_normalized(query, corpus)
    assert 0.9 < sim < 1.0


def test_normalize_makes_every_row_unit_length():
    raw = np.array(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    normalized = _normalize(raw)
    row_norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)


def test_normalize_preserves_direction():
    raw = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    assert np.allclose(_normalize(raw)[0], [0.6, 0.8, 0.0], atol=1e-5)


def test_normalize_zero_vector_does_not_produce_nan():
    # `+ 1e-9` in _normalize defends against div-by-zero on all-zero rows.
    # Without it, this row would become [nan, nan, nan, nan].
    raw = np.zeros((1, 4), dtype=np.float32)
    out = _normalize(raw)
    assert not np.any(np.isnan(out))
    assert np.allclose(out, 0.0)


def test_normalize_does_not_crash_on_empty_matrix():
    out = _normalize(np.zeros((0, 8), dtype=np.float32))
    assert out.shape == (0, 8)
