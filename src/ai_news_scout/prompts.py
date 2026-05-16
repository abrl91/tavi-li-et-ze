"""Versioned prompts. Bump VERSION when changing. Useful when comparing brief quality across runs."""

VERSION = "v3"


_STYLE_RULE = (
    "Style constraint: do NOT use em-dashes (the long dash, U+2014) or "
    "semicolons in your output. Use periods, commas, or restructure the "
    "sentence. Regular hyphens (-) are fine."
)


RANK_PROMPT = """You are an editor for an AI engineering Discord community.

Below are recent news/blog hits about "{topic}" pulled this week. Pick the TOP {k} by:
  1. Genuine novelty. Avoid items that rehash older news.
  2. Signal-to-noise. Prefer substance over hype or marketing.
  3. Discussion potential. Items that spark technical conversation among AI engineers.

Return STRICT JSON ONLY as a single object with key "picks" whose value is a list of {k} objects, each with keys "url" and "reason".
"reason" is one short sentence explaining the pick.
Output nothing else. No preamble, no code fences. Example shape:
{{"picks": [{{"url": "https://...", "reason": "..."}}, {{"url": "https://...", "reason": "..."}}]}}

""" + _STYLE_RULE + """

Items:
{items}
"""


WRITE_PROMPT = """You are writing one entry of a weekly "Explorer Brief" for an AI engineering Discord community. Audience: working ML/AI engineers and students.

Output exactly four lines in this format. The first line MUST be a Markdown link in the form `[title](url)` wrapped in `**...**`. Do NOT omit the URL or change the bracket/parenthesis order.

Worked example (do not copy the text. Copy the SHAPE):
**[Anthropic releases Claude Sonnet 4.6 with thinking](https://www.anthropic.com/news/claude-sonnet-4-6)**
*What it is:* Anthropic shipped a new Claude model with a built-in extended-thinking mode for complex reasoning.
*Why it matters:* Extended thinking lets agent loops trade latency for accuracy without bolt-on chain-of-thought scaffolding. Relevant if you're building anything multi-step.
*Discussion:* When does extended thinking pay off versus just running the cheaper model in a tool-use loop?

""" + _STYLE_RULE + """

Now write the entry for THIS source. Use the exact title and URL given below, do not paraphrase the title:

Source title: {title}
Source URL: {url}
Source content (your only ground truth. Do NOT invent facts not present here):
{content}

Write only the entry. Four lines, no preamble, no closing remarks.
"""
