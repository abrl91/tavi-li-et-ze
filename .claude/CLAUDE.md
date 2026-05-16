# Tavi-Li-Et-Ze (AI News Scout)

Tavily + Nebius Token Factory agent that scouts AI/ML news on a watchlist,
dedupes semantically via embeddings, ranks via LLM, and emits a four-line-entry
Markdown brief. Design doc: `docs/DESIGN.md`.

## Common commands

| Command | Purpose |
|---|---|
| `make smoke` | 1 Tavily + 1 chat + 1 embed call (~$0.0002) |
| `make brief-small` | Tiny pipeline run (1 topic, 3 results, top-2) |
| `make brief` | Full default run (5 topics × 10 results) |
| `make brief-commit` | Full run + git add+commit the brief file |
| `make api` | FastAPI viewer on :8000 |
| `make db` / `db-items` / `db-briefs` / `db-stats` | Inspect SQLite |
| `uv sync` | Install/update deps from `pyproject.toml` + `uv.lock` |

## Architecture

Eight-stage pipeline orchestrated in `pipeline.run()`:
**crawl** (Tavily per canonical source, optional) → **search** (Tavily per
topic) → **url-dedup** (within-run exact-match collapse across crawl + search)
→ **embed** (Nebius batch) → **novelty** (cosine vs prior corpus, threshold
0.86 — the ML core) → **rank** (one LLM call per topic, top-K JSON) →
**extract** (Tavily batch /extract on chosen URLs) → **write** (one LLM call
per chosen item).

URL dedup also has an across-run safety net: the `items` table declares
`url TEXT PRIMARY KEY` and `save_items` uses `INSERT OR IGNORE`, so a same-URL
twin slipping past cosine still gets dropped at the storage boundary.

`cli.py` and `api.py` are the **composition roots**: each reads `Settings()`,
constructs `Tavily()` + `Nebius()` + `Store()`, and hands them to
`pipeline.run()`. The pipeline never constructs providers itself — keeps it
stub-injectable for tests (see `tests/conftest.py`).

```
src/ai_news_scout/
  config.py     # Settings(BaseSettings) — env + .env
  providers.py  # Tavily, Nebius, SearchResult, Usage, PRICE_PER_MILLION_TOKENS_USD
  store.py      # SQLite: items+embeddings, briefs, cache
  prompts.py    # RANK_PROMPT, WRITE_PROMPT (versioned)
  profiler.py   # Profiler + StageStats + CacheStats — latency/tokens/$/cache + Tavily credits
  pipeline.py   # 8-stage run() with cached crawl + search + embeddings
  brief.py      # dict → Markdown renderer
  cli.py        # `brief` entry point (composition root)
  api.py        # FastAPI viewer with lifespan + Depends (composition root)
  _smoke.py     # provider sanity check, ~$0.0002, run before a full `brief`
```

## Conventions and policies

- **Always use the latest stable version** when adding or bumping a dep —
  `uv add <pkg>` auto-picks it; never guess from memory or LLM suggestion.
  See [`dependencies.md`](docs/dependencies.md).
- [`.claude/docs/conventions.md`](docs/conventions.md) — type system (Pydantic vs dataclass), naming, docstrings, composition/DI
- [`.claude/docs/dependencies.md`](docs/dependencies.md) — version pinning, `uv.lock`, latest-version policy
- [`.claude/docs/providers.md`](docs/providers.md) — Nebius / Tavily quirks and model-availability gotchas
