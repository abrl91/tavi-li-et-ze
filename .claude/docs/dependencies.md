# Dependencies

## Always use the latest stable version

When adding a new dep or bumping an existing one, fetch the **latest stable
release** from PyPI — do not guess from memory, do not copy from another
project, and do not trust an LLM's suggestion without verifying. Knowledge
cutoffs lag the package registry by months, and FastAPI / Pydantic / the
OpenAI SDK / Nebius / Tavily all ship breaking changes between minor
releases.

How to find the latest stable version:

- `uv add <pkg>` — auto-resolves to the latest non-pre-release and writes
  the pin into `pyproject.toml`. This is the preferred path.
- `uv pip index versions <pkg>` — lists every published version, newest
  first; stable releases have no `aN` / `bN` / `rcN` suffix.
- Or open `https://pypi.org/project/<pkg>/`.

"Stable" excludes pre-releases (`-alpha`, `-beta`, `-rc`, `-dev`); `uv add`
filters them out by default. Override only when you actually want one:
`uv add "fastapi==0.137.0a1"`.

## Pinning policy

- **All direct deps in `pyproject.toml` are pinned with `==`.** This makes the
  manifest auditable at a glance — every version sits in one file.
- **`uv.lock` is committed** for transitive pinning (`pydantic-core`, `httpx`,
  `anyio`, etc.). This is what reproduces an exact dependency graph across
  machines and CI.
- The combination is belt-and-suspenders: `pyproject.toml` locks direct deps
  even for someone running plain `pip install`; `uv.lock` locks everything for
  someone running `uv sync`.

## `python-dotenv` is kept explicitly

Even though only `pydantic-settings` calls it directly. The reason: in
`pydantic-settings >= 2`, `python-dotenv` is an *optional* dependency gated by
the `dotenv` extra. Listing it explicitly insulates us from a future
`pydantic-settings` release that drops the optional pull.

## Bumping a version

1. Edit `pyproject.toml` to the new exact version.
2. Run `uv sync`. This updates `uv.lock` and reinstalls.
3. Commit both files in the same change.

Don't edit `uv.lock` by hand — it's machine-generated. Don't use `uv lock
--upgrade` casually; it'll bump everything and produce a noisy diff.

## Trade-off

Pinning everything means a `numpy 2.4.5` perf fix doesn't reach you on the
next `uv sync` — you have to bump manually. For a single-app project where
audit-ability and reproducibility win, that's the right trade. For a library
you publish, prefer `>=` constraints so consumers don't conflict with their
other deps.
