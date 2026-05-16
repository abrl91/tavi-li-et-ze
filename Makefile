.PHONY: help sync smoke brief brief-small brief-debug brief-commit api api-prod db db-items db-briefs db-stats db-reset test clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---- run ----

sync:  ## install/update dependencies
	uv sync

smoke:  ## provider smoke test (1 Tavily + 1 LLM + 1 embed call, ~$0.0002)
	uv run python -m ai_news_scout._smoke

brief-small:  ## tiny pipeline run (1 topic, 3 results, top-2, ~$0.0005)
	uv run brief --topics "agentic search" --max-results 3 --top-k 2 --profile

brief-debug:  ## tiny run + print rendered RANK/WRITE prompts and LLM responses
	uv run brief --topics "agentic search" --max-results 3 --top-k 2 --profile --debug

brief:  ## full default run (5 topics x 10 results, profile on)
	uv run brief --profile

brief-commit:  ## full run + git add+commit the new brief file
	uv run brief --profile
	git add briefs/
	git commit -m "brief: $$(date +%F)"

test:  ## run the pytest suite (no network, no API spend)
	uv run pytest

api:  ## start the FastAPI viewer on :8000 (auto-reloads on code changes)
	@echo "→ AI News Scout viewer:  http://127.0.0.1:8000"
	uv run uvicorn ai_news_scout.api:app --host 127.0.0.1 --port 8000 --reload

api-prod:  ## same as `api` but no --reload (entry point from pyproject.toml)
	uv run api

# ---- inspect ----

db:  ## list tables with row counts
	@uv run sqlite-utils tables data/ai_news_scout.db --counts

db-items:  ## peek at items (skips the embedding BLOB)
	@uv run sqlite-utils data/ai_news_scout.db "SELECT url, title, topic, score, first_seen FROM items" --table

db-briefs:  ## list briefs with markdown size
	@uv run sqlite-utils data/ai_news_scout.db "SELECT date, length(markdown) AS md_chars, created_at FROM briefs ORDER BY date DESC" --table

db-stats:  ## raw stats JSON for the latest brief (pipe to | jq for pretty)
	@uv run sqlite-utils data/ai_news_scout.db "SELECT json(stats_json) FROM briefs ORDER BY date DESC LIMIT 1" --csv | tail -1

# ---- reset ----

db-reset:  ## wipe the SQLite DB (keeps committed briefs/)
	rm -f data/ai_news_scout.db

clean: db-reset  ## same as db-reset for now; reserved for broader cleanup later
