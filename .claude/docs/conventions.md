# Conventions

## Type system

| Use | When |
|---|---|
| `pydantic.BaseModel` | Data crosses a trust boundary (HTTP API, JSON file, user input). Validates and coerces at construction so bad data fails at the parse site, not 50 lines deeper inside the pipeline. |
| `pydantic_settings.BaseSettings` | Env vars + `.env` reading. **All env reads go through `Settings`** — never `os.environ[...]` in app code, never `load_dotenv()`. |
| `@dataclass` | Internal value objects you build yourself from already-trusted typed values (e.g., `Usage` from OpenAI SDK fields that are guaranteed `int`). |
| `typing.Literal[...]` | Fixed string sets that are API parameters, not domain concepts (e.g., `TavilyTopic`). No class needed; zero runtime cost. |

**Don't pydantic-ify everything for "consistency."** If there's no untrusted
input to validate, you pay ~10x construction overhead and an import for
nothing. The principle: pick the tool by *purpose*, not by uniformity.

A useful test: ask "could this object ever be constructed with bad data, and
where would I want the failure to happen?" If you can't think of a way bad
data gets in (because you're the only constructor), use `dataclass`. If bad
data is plausible (network, env, file), use Pydantic.

## Naming

- No single-letter names except math indices (`i`, `j`, `x`, `y`).
- `_client` not `_c`, `result` not `r`, `item` not `it`, `nebius` not `n`,
  `tavily` not `t`, `embeddings` not `mat`.
- Distribution name `tavi-li-et-ze` (hyphen) vs importable package
  `ai_news_scout` (underscore) — standard Python convention.

## Docstrings and comments

- **Module docstring**: 1–2 lines max.
- **Class docstring**: skip when the class name explains itself
  (`SearchResult`, `Usage`).
- **Method docstring**: Google-style — one sentence describing what it does,
  then `Args:`, then `Returns:`. No `Raises:` section: let library exceptions
  speak for themselves; documenting them here just rots when the SDK versions up.
- **Inline comments**: only when WHY is non-obvious (a workaround, a hidden
  constraint, a subtle invariant). Never describe WHAT — the code does that.
- Don't write design-rationale docstrings ("Pydantic was chosen because…").
  Either the choice still holds (comment is noise) or it doesn't (comment
  misleads). Design narrative belongs in PRs and `docs/`, not in code.

## Composition / dependency injection

- **`Tavily` and `Nebius` take explicit kwargs** in `__init__` — no `Settings`,
  no env. The CLI wires the values explicitly.
- **`pipeline.run()` requires `tavily` and `nebius` as kwargs** — no hidden
  default construction (no `tavily = tavily or Tavily()` shortcut). This is
  what makes the pipeline stub-injectable for tests.
- **`Settings()` is constructed once, in the CLI**, never elsewhere.
- Reading env at module top is forbidden — it captures values *before*
  any caller has a chance to populate `os.environ`. All env access goes through
  `Settings` (which uses `pydantic-settings` under the hood).
