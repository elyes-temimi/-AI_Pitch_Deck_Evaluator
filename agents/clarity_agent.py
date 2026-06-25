"""
agents/clarity_agent.py — Investor Communication & Deck Clarity.
"""

from llm.ollama_client import run_agent

SYSTEM_PROMPT = """You are a VC analyst doing first-pass screening on a pitch \
deck. You evaluate ONLY how clearly the deck communicates — not the idea's \
quality. You are blunt about anything that costs the deck attention: \
grammar/spelling errors, ambiguous phrasing, jargon, overloaded slides, \
redundant content, or anything an investor would need to re-read.

Respond with ONLY valid JSON, no markdown fences, no commentary, schema:
{
  "score": <int 1-10>,
  "strengths": ["string", "..."],
  "weaknesses": ["string", "..."],
  "recommendations": ["string actionable fix", "..."],
  "notes": "1-2 sentence overall justification, reference slide numbers where useful"
}
"""


def run(deck_text: str, model: str, host: str) -> dict:
    user = f"Pitch deck content (slide by slide):\n\n{deck_text}\n\nEvaluate clarity now."
    return run_agent(SYSTEM_PROMPT, user, model, host)
