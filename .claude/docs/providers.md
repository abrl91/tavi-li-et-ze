# Provider quirks

Footguns and non-obvious facts about Nebius Token Factory and Tavily —
documented here so they don't have to be rediscovered the hard way.

## Nebius Token Factory

- Productized successor to "Nebius AI Studio." **Use "Token Factory" in all
  narrative** (briefs, READMEs, comments). The keys console URL has been
  moving as the rebrand rolls out — don't hardcode it in `.env.example`.
- OpenAI-compatible: use the `openai` SDK with a custom `base_url` (held in
  `Settings.nebius_base_url`).
- Pricing per model lives in `providers.PRICE_PER_MILLION_TOKENS_USD`,
  co-located with the model IDs so a model swap is one diff.

### Model availability — known facts

| Model | Status | Notes |
|---|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | ✅ Default LLM | Good quality, ~10x cost vs 8B |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | ✅ Cheap option | Breaks Markdown links — `WRITE_PROMPT` includes a worked example to compensate |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ✅ Recommended rank | ~30% cheaper than Llama-3.3-70B, structured-output friendly. Set `RANK_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507` to route just the RANK stage to it. Default is still Llama-3.3-70B for both stages to preserve the quality baseline |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | ❌ **Does not exist** | 404. Don't be fooled by the naming pattern; use `Llama-3.3-70B-Instruct` |
| `Qwen/Qwen3-Embedding-8B` | ✅ Only embedding model | 4096-dim. Don't reach for `BAAI/bge-en-icl` — Nebius does not have it |

### Per-stage model routing

`Settings` exposes `rank_model` and `write_model`, both defaulting to
`nebius_llm_model`. `Nebius.chat(..., model=...)` accepts a per-call override
so the pipeline can route RANK and WRITE to different models from one client.
The composition root (`cli.py`, `api.py`) reads the settings and passes them
into `pipeline.run`.

### `response_format` and Token Factory enforcement

`pipeline._rank` passes `response_format={"type": "json_object"}` on the first
call. Token Factory enforces this as "top-level reply must be a JSON object."
RANK_PROMPT v3 was specifically written to ask for `{"picks": [...]}` so the
constraint is satisfied without distorting the model's output. With v2 (which
asked for a bare array), Llama-3.1-8B collapsed to a single `{url, reason}`
object to satisfy the constraint, breaking the parser. v3 parses first-try on
both `Llama-3.1-8B-Instruct` and `Llama-3.3-70B-Instruct` (verified
2026-05-16).

The retry fallback in `_rank` is still wired and gated on a `logging.WARNING`
("RANK retry: model=… returned non-JSON despite response_format=json_object").
With v3 the warning never fires, but the retry stays as a safety net in case a
future prompt revision reintroduces the shape mismatch. If the warning starts
firing, first hypothesis: someone changed the RANK_PROMPT shape and forgot
to keep it top-level-object.

If you suspect a new model has landed, list the catalog directly:

```python
from openai import OpenAI
client = OpenAI(api_key=..., base_url="https://api.tokenfactory.nebius.com/v1")
for m in client.models.list().data:
    print(m.id)
```

## Tavily

- Defaults that matter for an Explorer-style brief:
  - `topic="news"` — biases ranking toward news outlets.
  - `time_range="week"` — last 7 days only.
  - `search_depth="advanced"` — costs more credits per call but worth it for
    ~10 results per topic.
  - `include_raw_content="markdown"` — gets us article body as Markdown, not
    just a snippet. Free here; lets the LLM do better summarization later.
- **`topic_label` (our internal field on `SearchResult`) is not Tavily's
  `topic` param.** `topic` is a Tavily content bias (`news`/`general`/`finance`).
  `topic_label` is our watchlist label, attached to each result for downstream
  grouping. The two happen to share a name; don't conflate them.
