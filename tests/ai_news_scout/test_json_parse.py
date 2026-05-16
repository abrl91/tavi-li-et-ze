"""JSON-extraction helpers used by the RANK stage to parse model output."""

from __future__ import annotations

from ai_news_scout.pipeline import _parse_json_list, _strip_code_fence


def test_strip_code_fence_handles_bare_text():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_strip_code_fence_handles_lang_tagged_fence():
    text = '```json\n{"picks": []}\n```'
    assert _strip_code_fence(text).startswith("{")


def test_strip_code_fence_handles_plain_fence():
    text = '```\n[1, 2, 3]\n```'
    assert _strip_code_fence(text).startswith("[")


def test_parse_bare_list():
    assert _parse_json_list('[{"url": "u", "reason": "r"}]') == [
        {"url": "u", "reason": "r"}
    ]


def test_parse_dict_with_single_list_value_unwraps():
    # Production RANK_PROMPT asks for {"picks": [...]} so response_format=json_object
    # has a top-level object to satisfy. The parser unwraps it back to a list.
    out = _parse_json_list('{"picks": [{"url": "u1"}, {"url": "u2"}]}')
    assert out == [{"url": "u1"}, {"url": "u2"}]


def test_parse_fenced_object_with_picks():
    text = '```json\n{"picks": [{"url": "u"}]}\n```'
    assert _parse_json_list(text) == [{"url": "u"}]


def test_parse_prose_with_json_suffix_returns_none():
    assert _parse_json_list('Here you go: [{"url": "u"}]') is None


def test_parse_scalar_returns_none():
    assert _parse_json_list("42") is None


def test_parse_object_with_no_list_value_returns_none():
    assert _parse_json_list('{"reason": "no picks here"}') is None


def test_parse_object_with_multiple_list_values_returns_none():
    # Ambiguous: which list is the answer? Refuse rather than guess.
    assert _parse_json_list('{"picks": [1], "extras": [2]}') is None
