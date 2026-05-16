"""Settings loaded from env + .env (via pydantic-settings)."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Topic to query expansion. Keys are user-facing watchlist labels, values are
# the actual query strings sent to Tavily. Unmapped topics pass through verbatim.
# Goal: bare topic strings ("RAG eval") return noise; expanded queries
# ("retrieval augmented generation evaluation benchmarks") hit relevant news.
TOPIC_QUERIES_DEFAULT: dict[str, str] = {
    "agentic search":
        "agentic search retrieval AI agents production tools 2026",
    "RAG eval":
        "retrieval augmented generation evaluation benchmarks RAGAS",
    "AI inference":
        "LLM inference optimization vLLM TensorRT-LLM throughput latency",
    "MCP":
        "Model Context Protocol MCP server Anthropic integration",
    "small LLMs":
        "small language models open weights efficient on-device inference",
}


# Canonical sources for the crawl stage. Each entry is a vendor news/blog
# whose recent posts are reliably newsworthy without needing search. Crawled
# with one-hop depth so we pick up linked stories from the index page.
CANONICAL_SOURCES_DEFAULT: list[dict[str, str]] = [
    {
        "name": "anthropic-news",
        "url": "https://www.anthropic.com/news",
        "instructions": "Recent posts about model releases and research.",
    },
    {
        "name": "openai-news",
        "url": "https://openai.com/news/",
        "instructions": "Recent posts about model releases, API changes, and research.",
    },
    {
        "name": "mistral-news",
        "url": "https://mistral.ai/news",
        "instructions": "Recent posts about model releases and product news.",
    },
    {
        "name": "vllm-blog",
        "url": "https://blog.vllm.ai",
        "instructions": "Recent posts about inference performance and new features.",
    },
    {
        "name": "nebius-blog",
        "url": "https://nebius.com/blog",
        "instructions": "Recent posts about AI infrastructure and Token Factory updates.",
    },
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tavily_api_key: str = Field(min_length=1)
    nebius_api_key: str = Field(min_length=1)

    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    nebius_embed_model: str = "Qwen/Qwen3-Embedding-8B"

    # Per-stage model overrides. Unset means "use nebius_llm_model".
    # Set RANK_MODEL=... or WRITE_MODEL=... in env to route per stage.
    rank_model: str = ""
    write_model: str = ""

    # Override via env as a JSON array, e.g. DEFAULT_TOPICS='["agentic search","RAG eval"]'.
    default_topics: list[str] = Field(
        default_factory=lambda: [
            "agentic search",
            "RAG eval",
            "AI inference",
            "MCP",
            "small LLMs",
        ]
    )

    # Topic to query expansion. Override via env as JSON object, e.g.
    # TOPIC_QUERIES='{"agentic search":"agentic retrieval 2026"}'.
    topic_queries: dict[str, str] = Field(
        default_factory=lambda: dict(TOPIC_QUERIES_DEFAULT)
    )

    # Canonical sources crawled before search. Override via env as JSON array.
    canonical_sources: list[dict[str, str]] = Field(
        default_factory=lambda: [dict(s) for s in CANONICAL_SOURCES_DEFAULT]
    )

    search_exclude_domains: list[str] = Field(
        default_factory=lambda: ["medium.com", "dev.to", "substack.com"]
    )

    # Master switch for the crawl stage. Off avoids Tavily crawl credits when
    # iterating locally on the rest of the pipeline.
    enable_crawl: bool = True

    @model_validator(mode="after")
    def _default_stage_models(self) -> "Settings":
        if not self.rank_model:
            self.rank_model = self.nebius_llm_model
        if not self.write_model:
            self.write_model = self.nebius_llm_model
        return self
