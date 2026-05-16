"""Per-stage latency, token, and dollar tracking for the pipeline.

Use as a context manager. Token usage added inside the block is attributed to
the open stage; cache hits/misses are recorded on a separate ledger.

    profiler = Profiler()
    with profiler.stage("rank"):
        text, usage = nebius.chat(...)
        profiler.add_usage(usage)
    profiler.cache_event("embed", hit=True)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

from .providers import PRICE_PER_MILLION_TOKENS_USD, TAVILY_CREDIT_USD, Usage

_log = logging.getLogger(__name__)
_warned_models: set[str] = set()


@dataclass
class StageStats:
    seconds: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tavily_credits: float = 0.0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class Profiler:
    """Accumulates per-stage timings + token usage and per-cache hit/miss counts."""

    def __init__(self) -> None:
        self.stages: dict[str, StageStats] = {}
        self.caches: dict[str, CacheStats] = {}
        self._current: str | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Open a stage; usage added inside the block lands on this stage.

        Args:
            name: Stage label (e.g. `"search"`, `"rank"`). Re-entering the same
                name accumulates into the existing entry.
        """
        stats = self.stages.setdefault(name, StageStats())
        prev = self._current
        self._current = name
        t0 = time.perf_counter()
        try:
            yield
        finally:
            stats.seconds += time.perf_counter() - t0
            stats.calls += 1
            self._current = prev

    def add_usage(self, usage: Usage) -> None:
        """Attribute one LLM call's tokens + dollar cost to the current stage.

        Args:
            usage: `Usage` returned by `Nebius.chat` or `Nebius.embed`.
        """
        if self._current is None:
            return
        stats = self.stages[self._current]
        stats.tokens_in += usage.prompt_tokens
        stats.tokens_out += usage.completion_tokens
        stats.cost_usd += _cost_of(usage)

    def add_tavily_credits(self, credits: float) -> None:
        """Attribute Tavily API credits + their dollar equivalent to the current stage.

        Args:
            credits: Float number of credits the just-completed Tavily call
                consumed (e.g. `2.0` for one advanced search,
                `0.2 * len(urls)` for a basic extract batch).
        """
        if self._current is None:
            return
        stats = self.stages[self._current]
        stats.tavily_credits += credits
        stats.cost_usd += credits * TAVILY_CREDIT_USD

    def cache_event(self, name: str, *, hit: bool) -> None:
        """Record one cache hit or miss for the named cache.

        Args:
            name: Cache label (e.g. `"search"`, `"embed"`).
            hit: True for a hit, False for a miss.
        """
        cs = self.caches.setdefault(name, CacheStats())
        if hit:
            cs.hits += 1
        else:
            cs.misses += 1

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of all stats so far."""
        return {
            "stages": {n: asdict(s) for n, s in self.stages.items()},
            "caches": {
                n: {"hits": c.hits, "misses": c.misses, "hit_rate": c.hit_rate}
                for n, c in self.caches.items()
            },
            "totals": {
                "seconds": sum(s.seconds for s in self.stages.values()),
                "tokens_in": sum(s.tokens_in for s in self.stages.values()),
                "tokens_out": sum(s.tokens_out for s in self.stages.values()),
                "cost_usd": sum(s.cost_usd for s in self.stages.values()),
                "tavily_credits": sum(s.tavily_credits for s in self.stages.values()),
            },
        }

    def pretty_table(self) -> str:
        """Render a human-readable table for the CLI `--profile` flag."""
        lines: list[str] = []
        header = (
            f"  {'stage':<10} {'calls':>6} {'sec':>7} {'tok_in':>9} {'tok_out':>9} "
            f"{'credits':>8} {'$':>9}"
        )
        rule = "  " + "─" * (len(header) - 2)
        lines.append(header)
        lines.append(rule)
        for name, s in self.stages.items():
            lines.append(
                f"  {name:<10} {s.calls:>6} {s.seconds:>7.2f} "
                f"{s.tokens_in:>9} {s.tokens_out:>9} "
                f"{s.tavily_credits:>8.1f} {s.cost_usd:>9.4f}"
            )
        totals = self.to_dict()["totals"]
        lines.append(rule)
        lines.append(
            f"  {'total':<10} {'':>6} {totals['seconds']:>7.2f} "
            f"{totals['tokens_in']:>9} {totals['tokens_out']:>9} "
            f"{totals['tavily_credits']:>8.1f} {totals['cost_usd']:>9.4f}"
        )
        if self.caches:
            lines.append("")
            lines.append(f"  {'cache':<10} {'hits':>6} {'misses':>6} {'hit_rate':>9}")
            lines.append("  " + "─" * 36)
            for name, c in self.caches.items():
                lines.append(
                    f"  {name:<10} {c.hits:>6} {c.misses:>6} {c.hit_rate * 100:>8.1f}%"
                )
        return "\n".join(lines)


def _cost_of(usage: Usage) -> float:
    """USD cost of one chat/embed call. Returns 0.0 when the model is unpriced."""
    prices = PRICE_PER_MILLION_TOKENS_USD.get(usage.model)
    if prices is None:
        if usage.model not in _warned_models:
            _warned_models.add(usage.model)
            _log.warning(
                "no price entry for model %r, cost will be reported as $0.00",
                usage.model,
            )
        return 0.0
    return (
        usage.prompt_tokens * prices["in"] + usage.completion_tokens * prices["out"]
    ) / 1_000_000
