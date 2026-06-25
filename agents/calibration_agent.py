"""
agents/calibration_agent.py — Investor Calibration Requirement (the
assignment's section 4), implemented as REAL retrieval-augmented
generation, not a static prompt block.

How it actually works:
1. Pull the specific text of each relevant slide from THIS deck (Problem,
   Solution, Market, Traction, Team, Ask) using extraction.get_section.
2. For each, retrieve the most relevant benchmark comparables from
   rag/benchmark_data.py via rag/retrieval.py's TF-IDF search — so a
   hardware deck's solution slide retrieves hardware-relevant anchors, a
   fintech deck's problem slide retrieves fintech-relevant anchors, etc.
3. Feed the model ONLY the comparables that were actually retrieved for
   THIS deck (with their tier and id), and ask it to explicitly compare
   the deck's content against them, citing which anchor id it's using.
4. The agent must say when no good comparable was retrieved (low
   similarity) rather than forcing a comparison — this keeps the
   calibration honest instead of hallucinating relevance.

This is the piece that makes the "investor calibration" claim true rather
than asserted: you can point at retrieval.py and show, live, that two
different decks retrieve two different sets of comparables.
"""

import json

from llm.ollama_client import run_agent
from extraction.extractor import get_section
from rag.retrieval import retrieve_for_deck

SYSTEM_PROMPT = """You are the Investor Calibration Agent on a VC diligence \
team. You do not score clarity, narrative, or problem-solution fit \
yourself — those are handled by other analysts. Your ONLY job is to \
compare this specific deck's content against a set of RETRIEVED reference \
characteristics (each tagged with an id and a tier: exceptional / average / \
weak), and judge how this deck's actual content compares to each one.

Rules:
- Only comment on a dimension if at least one reference was retrieved for it.
  If none were retrieved (empty list), say so explicitly rather than inventing a comparison.
- Always cite the reference id(s) you are comparing against (e.g. "vs PF01").
- A LOW similarity score on a retrieved reference means it's a weak match —
  say so honestly rather than forcing the comparison; do not overstate confidence.
- Be specific: reference the deck's own wording where relevant, not generic praise/criticism.
- You will be told the deck's inferred FUNDING STAGE (pre-seed/seed/series-a/series-b).
  Scale your expectations accordingly: light detail on GTM, formal business
  model mechanics, or financials is NORMAL and should not be flagged as a
  gap for a pre-seed or seed deck — only flag it as a gap if it's missing
  even by the standard of that stage. A pre-seed deck is judged on whether
  the problem/solution/market case is compelling, not on operational maturity.
  The retrieved references for traction/market_sizing/ask/business_model
  have ALREADY been filtered to match this deck's stage, so trust that the
  references shown to you are stage-appropriate — do not second-guess
  whether a retrieved traction anchor is "too lenient" or "too strict" for
  the stage, that filtering already happened upstream.
- A problem can match an "exceptional" pattern either by citing a specific
  sourced number, OR by being so universally self-evident that a reader
  doesn't need a statistic to find it credible (e.g. "hotels are expensive").
  Do not penalize the second case just because it lacks a number — check
  whether a self-evident-pain anchor was retrieved before concluding "weak".
- If the retrieved references for a dimension disagree with what the deck
  actually demonstrates (e.g. a short, simple sentence is retrieved against
  a "weak/thin" anchor but the brevity is clearly deliberate and effective),
  use your own judgment over a forced literal match — note this explicitly
  rather than mechanically deferring to the closest retrieved anchor.

Respond with ONLY valid JSON, no markdown fences, no commentary, schema:
{
  "benchmark_alignment_score": <int 1-10, overall alignment with exceptional-tier patterns across all retrieved dimensions, adjusted for stage>,
  "per_dimension": {
    "<dimension_name>": {
      "comparison": "string, 1-2 sentences citing the reference id(s) used",
      "alignment": "string, one of: 'matches exceptional pattern', 'matches average pattern', 'matches weak pattern', 'no strong reference match'"
    }
  },
  "standout_strengths": ["string, things that match exceptional-tier patterns", "..."],
  "gaps_vs_benchmark": ["string, things that match average/weak-tier patterns, excluding anything explained by stage", "..."]
}
"""


def gather_retrieval_context(slides, deck_text: str, industry: str = None, stage: str = None, top_k: int = 2):
    """Builds the per-dimension query text from THIS deck's actual slides,
    then retrieves comparables for each. Returns the retrieval results dict
    (dimension -> list of retrieved snippets).

    `stage` is passed through to retrieval so the FILTERING happens here,
    not just as an instruction in the prompt: for stage-sensitive dimensions
    (traction, market_sizing, ask, business_model), retrieve() restricts the
    candidate pool to anchors tagged for this deck's actual stage (or
    stage-agnostic 'all' anchors) before ranking by similarity."""
    queries = {
        "problem_framing": get_section(slides, "problem", "pain"),
        "solution_clarity": get_section(slides, "solution", "product"),
        "market_sizing": get_section(slides, "market", "opportunity", "tam"),
        "traction": get_section(slides, "traction", "signal", "metrics", "validation", "early signal"),
        "business_model": get_section(slides, "business model", "revenue", "monetization", "pricing"),
        "team": get_section(slides, "team", "founders"),
        "ask": get_section(slides, "ask", "raise", "funding"),
        "narrative_flow": deck_text[:2000],   # whole-deck signal, truncated for the query
        "clarity_writing": deck_text[:2000],
    }
    return retrieve_for_deck(queries, industry=industry, stage=stage, top_k=top_k)


def _format_retrieved_for_prompt(retrieved: dict) -> str:
    lines = []
    for dim, items in retrieved.items():
        if not items:
            lines.append(f"\n[{dim}] -> no reference retrieved (no matching slide content found)")
            continue
        lines.append(f"\n[{dim}]")
        for it in items:
            lines.append(f"  - {it['id']} (tier: {it['tier']}, similarity: {it['similarity']}): {it['text']}")
    return "\n".join(lines)


def run(slides, deck_text: str, model: str, host: str, industry: str = None,
        stage: str = "seed", top_k: int = 3):
    retrieved = gather_retrieval_context(slides, deck_text, industry=industry, stage=stage, top_k=top_k)
    retrieved_block = _format_retrieved_for_prompt(retrieved)

    user = (
        f"DECK'S INFERRED FUNDING STAGE: {stage}\n\n"
        f"DECK CONTENT (slide by slide):\n\n{deck_text}\n\n"
        f"RETRIEVED REFERENCE CHARACTERISTICS FOR THIS DECK:\n{retrieved_block}\n\n"
        "Compare this deck's actual content against the retrieved references above, "
        "dimension by dimension, citing reference ids, and scaling your expectations "
        "to the stated funding stage. Produce your calibration verdict now."
    )
    result = run_agent(SYSTEM_PROMPT, user, model, host)
    # Attach the raw retrieval trace so the report/UI can show exactly what
    # was retrieved for this deck — this is the proof that it's real RAG.
    result["_retrieval_trace"] = retrieved
    return result