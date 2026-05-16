"""`brief` CLI entry point — composition root for env, providers, and pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .pipeline import run as run_pipeline
from .providers import Nebius, Tavily


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    tavily = Tavily(api_key=settings.tavily_api_key)
    nebius = Nebius(
        api_key=settings.nebius_api_key,
        llm_model=settings.nebius_llm_model,
        embed_model=settings.nebius_embed_model,
        base_url=settings.nebius_base_url,
    )

    parser = argparse.ArgumentParser(prog="brief", description="Generate the weekly Explorer Brief.")
    parser.add_argument("--topics", default=None, help=f"comma-separated watchlist (default: {','.join(settings.default_topics)})")
    parser.add_argument("--max-results", type=int, default=10, help="Tavily results per topic (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=3, help="items selected per topic (default: %(default)s)")
    parser.add_argument("--dedup-threshold", type=float, default=0.86, help="max cosine vs prior items before drop (default: %(default)s)")
    parser.add_argument("--no-crawl", action="store_true", help="skip the Tavily crawl stage (saves credits)")
    parser.add_argument("--profile", action="store_true", help="print the per-stage profile table")
    parser.add_argument("--debug", action="store_true", help="print rendered RANK/WRITE prompts and LLM responses")
    args = parser.parse_args(argv)

    if args.topics is None:
        topics = list(settings.default_topics)
    else:
        topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    print(f"→ topics: {topics}")
    result = run_pipeline(
        topics,
        tavily=tavily,
        nebius=nebius,
        dedup_threshold=args.dedup_threshold,
        max_results_per_topic=args.max_results,
        top_k_per_topic=args.top_k,
        rank_model=settings.rank_model,
        write_model=settings.write_model,
        topic_queries=settings.topic_queries,
        canonical_sources=settings.canonical_sources,
        search_exclude_domains=settings.search_exclude_domains,
        enable_crawl=settings.enable_crawl and not args.no_crawl,
        debug=args.debug,
    )

    out_path = Path("briefs") / f"{result['date']}.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result["markdown"])
    print(f"\n✓ wrote {out_path}  ({result['stats']['n_in_brief']} entries)")

    if args.profile:
        stats = result["stats"]
        print("\n- pipeline stats -")
        print(f"  searched         : {stats['n_searched']}")
        print(f"  after url-dedup  : {stats['n_after_url_dedup']}")
        print(f"  after novelty    : {stats['n_after_novelty']}  (dropped {stats['n_dropped_novelty']})")
        print(f"  in brief         : {stats['n_in_brief']}")
        print()
        print(result["profiler"].pretty_table())


if __name__ == "__main__":
    main()
