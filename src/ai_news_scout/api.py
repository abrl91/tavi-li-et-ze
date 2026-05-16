"""FastAPI viewer for Explorer Briefs.

Routes:
  GET  /                    -> latest brief + sidebar + stats footer
  GET  /briefs/{date}       -> historical brief (same template)
  POST /run                 -> schedules a pipeline run, returns 202 + {status, date}
  GET  /runs/{date}/status  -> reports whether a brief for {date} has landed

The Markdown brief is JSON-encoded into a <script> tag and rendered client-side
with marked.js (JSON transport avoids HTML-entity escaping inside <script>).
Styling is inline CSS via design-token custom properties, no Tailwind. The
`_normalize_stats` shim flattens older and newer stats shapes so any historical
DB row renders correctly.

Resources (Settings, Store, Tavily, Nebius) are constructed once in `lifespan`
and stashed on `app.state`. This keeps module import side-effect-free, so
`python -c "import ai_news_scout.api"` works without `.env` present, and lets
tests override providers via `app.dependency_overrides`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date as date_cls
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .pipeline import run as run_pipeline
from .providers import Nebius, Tavily
from .store import Store


_log = logging.getLogger(__name__)
_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    store = Store()
    app.state.settings = settings
    app.state.store = store
    app.state.tavily = Tavily(api_key=settings.tavily_api_key)
    app.state.nebius = Nebius(
        api_key=settings.nebius_api_key,
        llm_model=settings.nebius_llm_model,
        embed_model=settings.nebius_embed_model,
        base_url=settings.nebius_base_url,
    )
    try:
        yield
    finally:
        store._conn.close()


app = FastAPI(title="Explorer Brief", lifespan=lifespan)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_tavily(request: Request) -> Tavily:
    return request.app.state.tavily


def get_nebius(request: Request) -> Nebius:
    return request.app.state.nebius


def _normalize_stats(stats: dict) -> dict[str, Any]:
    """Flatten new- and old-shape stats into one dict the template can read."""
    profiler = stats.get("profiler") or {}
    totals = profiler.get("totals") or {}
    return {
        "n_in_brief": stats.get("n_in_brief", 0),
        "n_searched": stats.get("n_searched", 0),
        "n_after_url_dedup": stats.get("n_after_url_dedup", 0),
        "n_after_novelty": stats.get("n_after_novelty", 0),
        "n_dropped_novelty": stats.get("n_dropped_novelty", 0),
        "total_seconds": totals.get("seconds", stats.get("total_seconds", 0.0)),
        "tokens_in": totals.get("tokens_in", 0),
        "tokens_out": totals.get("tokens_out", 0),
        "cost_usd": totals.get("cost_usd", 0.0),
        "stages": profiler.get("stages") or {},
        "caches": profiler.get("caches") or {},
    }


def _render(request: Request, store: Store, brief: dict | None) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"brief": brief, "briefs": store.list_briefs()},
    )


@app.get("/", response_class=HTMLResponse)
def root(
    request: Request,
    store: Store = Depends(get_store),
) -> HTMLResponse:
    latest = store.latest_brief()
    if latest is None:
        return _render(request, store, brief=None)
    date, markdown, stats = latest
    return _render(
        request,
        store,
        brief={"date": date, "markdown": markdown, "stats": _normalize_stats(stats)},
    )


@app.get("/briefs/{date}", response_class=HTMLResponse)
def brief_by_date(
    request: Request,
    date: str,
    store: Store = Depends(get_store),
) -> HTMLResponse:
    got = store.get_brief(date)
    if got is None:
        raise HTTPException(status_code=404, detail=f"no brief for {date}")
    markdown, stats = got
    return _render(
        request,
        store,
        brief={"date": date, "markdown": markdown, "stats": _normalize_stats(stats)},
    )


def _run_pipeline_bg(
    settings: Settings,
    tavily: Tavily,
    nebius: Nebius,
    store: Store,
) -> None:
    try:
        run_pipeline(
            settings.default_topics,
            tavily=tavily,
            nebius=nebius,
            store=store,
            rank_model=settings.rank_model,
            write_model=settings.write_model,
            topic_queries=settings.topic_queries,
            canonical_sources=settings.canonical_sources,
            search_exclude_domains=settings.search_exclude_domains,
            enable_crawl=settings.enable_crawl,
        )
    except Exception:
        # Background tasks have no caller to raise to. Log so the failure is
        # observable in stderr instead of disappearing silently.
        _log.exception("background pipeline run failed")


@app.post("/run", status_code=202)
def trigger_run(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    tavily: Tavily = Depends(get_tavily),
    nebius: Nebius = Depends(get_nebius),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    today = date_cls.today().isoformat()
    background_tasks.add_task(_run_pipeline_bg, settings, tavily, nebius, store)
    return {"status": "started", "date": today}


@app.get("/runs/{date}/status")
def run_status(
    date: str,
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    return {"date": date, "ready": store.get_brief(date) is not None}


def run() -> None:
    """`api` script entry point (see `[project.scripts]` in pyproject.toml)."""
    uvicorn.run("ai_news_scout.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
