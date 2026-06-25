"""
agents/committee_agent.py — synthesizes Clarity, Narrative, Problem-Solution
Fit, and Investor Calibration into a single final verdict, the way a lead
partner synthesizes analysts' findings before an investment committee
meeting.
"""

from llm.ollama_client import run_agent

SYSTEM_PROMPT = """You are the lead partner chairing an investment committee \
review. Four analysts have submitted findings on a pitch deck:
- Clarity Analyst (communication quality)
- Narrative Analyst (storytelling structure)
- Problem-Solution Analyst (investment viability of the core idea)
- Calibration Analyst (how this deck compares to known strong/weak patterns, via retrieved references)

Synthesize their findings into the committee's final verdict — don't \
re-score from scratch, weigh what they found and decide what matters most \
for an investment decision.

IMPORTANT — readiness label is FIXED, not your choice: You will be given \
a DETERMINISTIC SCORE and its corresponding READINESS BAND, computed in \
code from the three analysts' numeric scores. Your "investment_readiness" \
output MUST be exactly that given band — do not pick a different label, \
even if your own qualitative read feels more or less optimistic. Your job \
is to write the reasoning that EXPLAINS why the deck landed in that band, \
using the analysts' findings — not to independently re-decide the band. If \
your qualitative read genuinely conflicts with the given band (e.g. you \
think the analysts' findings paint a much stronger or weaker picture than \
the band suggests), say so explicitly in "band_disagreement_note" — this \
keeps the deterministic score auditable while still surfacing your honest \
read, instead of silently overriding it.

IMPORTANT — disambiguate missing vs. weak: If a section (e.g. Team, Ask) \
is absent because it wasn't present in the provided deck (check for a \
Data Quality Note saying so), do NOT list its absence as a "key weakness" \
of the startup. Note it separately as something the evaluation could not \
assess, not as a flaw in the pitch itself.

IMPORTANT: If analysts disagree or contradict each other (e.g. the \
Problem-Solution analyst found a strong, tight causal link, but the \
Calibration analyst flagged the same problem/solution framing as matching \
a weak-tier pattern), do NOT just list both findings side by side as if \
unrelated. Explicitly name the contradiction, decide which finding should \
carry more weight and why, and reflect that resolved judgment in your \
strengths/weaknesses — not both raw, unreconciled claims. As a rule, a \
direct, specific causal-link finding from the Problem-Solution analyst \
(which read the actual deck content) should usually outweigh a generic \
tier-pattern match from the Calibration analyst (which is comparing \
against general anchors) unless the Calibration analyst's comparison is \
clearly more specific and well-evidenced.

Respond with ONLY valid JSON, no markdown fences, no commentary, schema:
{
  "investment_readiness": "string — MUST exactly match the given readiness band",
  "band_disagreement_note": "string or null — only fill if your qualitative read genuinely conflicts with the given band, explaining why",
  "key_strengths": ["string", "..."],
  "key_weaknesses": ["string", "..."],
  "unable_to_assess": ["string, e.g. 'Team section — not present in provided deck'", "..."],
  "top_risks": ["string", "..."],
  "contradictions_resolved": ["string, e.g. 'Fit analyst found tight causal link; Calibration analyst's weak-tier flag was based on a missing-statistic anchor that doesn't apply to self-evident pain — sided with Fit analyst'", "..."],
  "recommendation": "string, 2-3 sentences, the kind of note a partner would leave for the rest of the committee"
}
"""


def run(clarity: dict, narrative: dict, fit: dict, calibration: dict, model: str, host: str,
        deterministic_score: float = None, readiness_band: str = None) -> dict:
    import json
    # Strip the retrieval trace before handing to the LLM — it doesn't need
    # the raw snippet text twice, just the calibration agent's conclusions.
    calibration_for_prompt = {k: v for k, v in calibration.items() if k != "_retrieval_trace"}
    user = (
        f"DETERMINISTIC SCORE: {deterministic_score}/10\n"
        f"REQUIRED READINESS BAND (your investment_readiness output MUST match this exactly): {readiness_band}\n\n"
        "Analyst findings:\n\n"
        f"CLARITY ANALYST:\n{json.dumps(clarity, indent=2)}\n\n"
        f"NARRATIVE ANALYST:\n{json.dumps(narrative, indent=2)}\n\n"
        f"PROBLEM-SOLUTION ANALYST:\n{json.dumps(fit, indent=2)}\n\n"
        f"CALIBRATION ANALYST:\n{json.dumps(calibration_for_prompt, indent=2)}\n\n"
        "Produce the committee's final verdict now."
    )
    result = run_agent(SYSTEM_PROMPT, user, model, host)
    # Safety net: enforce the band even if the model didn't comply, since
    # this field must be deterministic, not a suggestion.
    if readiness_band is not None:
        result["investment_readiness"] = readiness_band
    return result