"""Provider sanity check: 1 Tavily search + 1 chat + 1 embed call, ~$0.0002.

Useful before running the full pipeline to confirm both API keys are valid.

Run: `uv run python -m ai_news_scout._smoke`
"""

from __future__ import annotations

from ai_news_scout.config import Settings
from ai_news_scout.providers import Nebius, Tavily


def main() -> None:
    settings = Settings()

    print("→ Tavily")
    tavily = Tavily(api_key=settings.tavily_api_key)
    hits = tavily.search(
        "agentic search Tavily Nebius", topic_label="smoke", max_results=1
    )
    if hits:
        print(f"  ok - {len(hits)} result(s); first: {hits[0].title!r}")
    else:
        print("  ok - 0 results")

    print("→ Nebius LLM")
    nebius = Nebius(
        api_key=settings.nebius_api_key,
        llm_model=settings.nebius_llm_model,
        embed_model=settings.nebius_embed_model,
        base_url=settings.nebius_base_url,
    )
    text, usage = nebius.chat(
        [{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=10,
        temperature=0.0,
    )
    print(
        f"  ok - model={usage.model} tokens(in/out)={usage.prompt_tokens}/{usage.completion_tokens} reply={text.strip()!r}"
    )

    print("→ Nebius embeddings")
    embeddings, usage = nebius.embed(
        ["agentic search is a way to give LLMs fresh, ranked context"]
    )
    print(
        f"  ok - model={usage.model} shape={embeddings.shape} tokens={usage.prompt_tokens}"
    )

    print("\nALL GOOD ✓")


if __name__ == "__main__":
    main()
