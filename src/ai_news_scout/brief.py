"""Render assembled entries to the Markdown brief."""

from __future__ import annotations


_HEADER = """# Explorer Brief · {date}

"""


def render(
    entries: list[dict],
    for_date: str,
    *,
    n_searched: int = 0,
    n_dropped_novelty: int = 0,
) -> str:
    if entries:
        body = "\n\n---\n\n".join(e["markdown"].strip() for e in entries)
    else:
        body = _empty_body(n_searched=n_searched, n_dropped=n_dropped_novelty)
    return _HEADER.format(date=for_date) + body + "\n"


def _empty_body(*, n_searched: int, n_dropped: int) -> str:
    if n_searched == 0:
        return "_No items. The search step returned no results._"
    if n_dropped == 0:
        return (
            f"_No items. Searched **{n_searched}** candidates but the ranker "
            f"selected none._"
        )
    return (
        f"_No new items today. Searched **{n_searched}** candidates, "
        f"novelty filter dropped **{n_dropped}** as duplicates of items from "
        f"prior runs. The dedup ML is doing its job. A re-run right after a "
        f"previous run will always be empty._"
    )
