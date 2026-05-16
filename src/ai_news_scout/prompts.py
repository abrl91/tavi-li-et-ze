"""Versioned prompts.

Bump VERSION when any of these change so brief quality can be compared across
runs. Each stage uses a system message for role and rules plus a user message
for the task and data. The RANK response shape is defined as a Pydantic model
in `pipeline.py` (`RankResponse`) and enforced via the OpenAI SDK's
`chat.completions.parse()` which converts the model to a strict JSON schema.
"""

VERSION = "v4"


_STYLE_RULE = (
    "Do not use em-dashes (the long dash, U+2014) or semicolons. "
    "Use periods, commas, or restructure the sentence. "
    "Regular hyphens (-) are fine."
)


RANK_SYSTEM = (
    "You are an editor for an AI engineering community, selecting the "
    "best recent news and blog items for a weekly brief.\n\n"
    "Selection criteria, in order of importance:\n"
    "  1. Genuine novelty. Avoid items that rehash older news.\n"
    "  2. Signal-to-noise. Prefer substance over hype or marketing.\n"
    "  3. Discussion potential. Pick items that spark technical "
    "conversation among AI engineers.\n\n"
    "Style: " + _STYLE_RULE
)


RANK_USER = """Pick the TOP {k} items about "{topic}" from the list below.

For each pick, return the URL exactly as provided and one short sentence explaining why it made the cut.

Items:
{items}
"""


WRITE_SYSTEM = (
    "You are writing one entry of a weekly Explorer Brief for an AI "
    "engineering community. Audience: working ML and AI engineers and "
    "students.\n\n"
    "Output exactly four lines. No preamble, no closing remarks.\n\n"
    "Line 1 MUST be a bold Markdown link in this exact form:\n"
    "  **[TITLE](URL)**\n"
    "The double-asterisks wrap a Markdown link whose label is in square "
    "brackets [] and target is in round brackets () with no space between "
    "them. Wrong forms include `**TITLE** (URL)` (parens after bold, not a "
    "real link) and `**TITLE**` with no URL.\n\n"
    "Lines 2-4 use this structure (do not copy the example text, copy the "
    "structure):\n"
    "  *What it is:* One factual sentence describing the item.\n"
    "  *Why it matters:* One sentence on the technical or strategic implication.\n"
    "  *Discussion:* One open question relevant to AI engineers.\n\n"
    "Worked example:\n"
    "**[Example title goes here](https://example.com/sample-article)**\n"
    "*What it is:* A factual one-sentence description.\n"
    "*Why it matters:* The implication for engineers.\n"
    "*Discussion:* An open question.\n\n"
    "Rules:\n"
    "  - Use the exact title and URL provided. Do not paraphrase the title.\n"
    "  - Ground every claim in the source content provided. Do not invent "
    "facts that are not in the source.\n\n"
    "Style: " + _STYLE_RULE
)


WRITE_USER = """Source title: {title}
Source URL: {url}
Source content (your only ground truth):
{content}
"""
