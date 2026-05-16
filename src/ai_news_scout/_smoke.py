"""Provider sanity check: 1 Tavily search + 1 chat + 1 chat_parsed + 1 embed call, ~$0.0003.

Useful before running the full pipeline to confirm both API keys are valid
and that the structured-output (`chat_parsed`) path is supported by the
current Nebius model.

Run: `uv run python -m ai_news_scout._smoke`
"""

from __future__ import annotations

from pydantic import BaseModel

from ai_news_scout.config import Settings
from ai_news_scout.providers import Nebius, Tavily


class _SmokeReply(BaseModel):
    """Tiny schema used to verify Nebius accepts response_format=json_schema."""

    reply: str


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

    print("→ Nebius structured output (json_schema)")
    parsed, usage = nebius.chat_parsed(
        [
            {"role": "system", "content": "You return a single field named 'reply'."},
            {"role": "user", "content": "Set reply to 'pong'."},
        ],
        _SmokeReply,
        max_tokens=30,
        temperature=0.0,
    )
    print(
        f"  ok - model={usage.model} tokens(in/out)={usage.prompt_tokens}/{usage.completion_tokens} parsed={parsed!r}"
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
