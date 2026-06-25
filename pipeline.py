"""
pipeline.py
Orchestrates the full multi-agent, RAG-calibrated evaluation:

  1. Clarity Agent
  2. Narrative Agent
  3. Problem-Solution Fit Agent          (also guesses industry/stage, used by #4)
  4. Investor Calibration Agent (RAG)    (retrieves comparables FOR THIS DECK,
                                           compares against them)
  --- deterministic score computed here, BEFORE the committee runs ---
  5. Committee Agent                     (synthesizes 1-4 + the deterministic
                                           score into a final verdict whose
                                           readiness label is REQUIRED to
                                           match the score's band)

The headline 0-10 score is computed deterministically in Python from the
three core dimension scores (clarity/narrative/fit) — kept out of the LLM's
hands so it's reproducible. The investment-readiness LABEL is also pinned
deterministically from that score via READINESS_BANDS, specifically because
testing showed the committee agent could otherwise produce a verdict like
"Strong - fundable now" alongside a 7.2/10 score with no connection between
the two — the label and the number must come from the same source of truth.
Benchmark Alignment (from the Calibration Agent) is reported alongside as a
separate, transparent number rather than silently folded into the headline
score, since it measures something different (similarity to known patterns,
not the dimension itself).
"""

from agents import clarity_agent, narrative_agent, fit_agent, calibration_agent, committee_agent

WEIGHTS = {"clarity": 0.25, "narrative": 0.30, "fit": 0.45}

# Deterministic score -> readiness label, lowest threshold first.
READINESS_BANDS = [
    (8.5, "Exceptional - high priority"),
    (7.5, "Strong - fundable now"),
    (6.5, "Promising - take a meeting"),
    (5.0, "Needs work - pass for now"),
    (0.0, "Not investable yet"),
]


def compute_overall_score(clarity_score, narrative_score, fit_score) -> float:
    overall = (
        clarity_score * WEIGHTS["clarity"]
        + narrative_score * WEIGHTS["narrative"]
        + fit_score * WEIGHTS["fit"]
    )
    return round(overall, 2)


def score_to_readiness_band(score_10: float) -> str:
    for threshold, label in READINESS_BANDS:
        if score_10 >= threshold:
            return label
    return READINESS_BANDS[-1][1]


def _build_data_quality_block(extraction_warnings) -> str:
    """Turns extraction-layer warnings (stripped boilerplate, dropped promo
    slides, Lorem Ipsum flags, thin-content flags) into a context block
    every agent sees BEFORE evaluating. Without this, agents have no way to
    know a "grammar error" they're looking at is actually a PDF extraction
    artifact, or that a missing Team slide is a source-document gap rather
    than evidence the founders skipped it."""
    if not extraction_warnings:
        return ""
    lines = "\n".join(f"- {w}" for w in extraction_warnings)
    return (
        "\n\n=== DATA QUALITY NOTES (about THIS EXTRACTED COPY of the deck, "
        "not the original pitch) ===\n"
        f"{lines}\n"
        "Do NOT count any issue named above against the deck's communication "
        "quality, narrative completeness, or any other dimension — these are "
        "artifacts of converting the source file to text, not flaws in the "
        "founders' actual work. If a section is absent and not explained by "
        "a note above, say so as 'not present in the provided deck — cannot "
        "assess' rather than treating the absence itself as a weakness.\n"
        "=== END DATA QUALITY NOTES ===\n"
    )


def evaluate_deck(slides, deck_text: str, model: str = "llama3.1",
                   host: str = "http://localhost:11434", progress_cb=None,
                   extraction_warnings=None):
    """Runs the full 5-agent pipeline. progress_cb(msg) is called before each
    agent runs (used by the CLI/UI to show progress, since each call can
    take 30s-2min on a local model). extraction_warnings (from
    extraction.extractor.extract_deck) are folded into every agent's input
    as a data-quality context block — see _build_data_quality_block."""

    def tick(msg):
        if progress_cb:
            progress_cb(msg)

    quality_block = _build_data_quality_block(extraction_warnings)
    deck_text_for_agents = deck_text + quality_block

    tick("Running Clarity Agent...")
    clarity = clarity_agent.run(deck_text_for_agents, model, host)

    tick("Running Narrative Agent...")
    narrative = narrative_agent.run(deck_text_for_agents, model, host)

    tick("Running Problem-Solution Agent...")
    fit = fit_agent.run(deck_text_for_agents, model, host)

    tick("Retrieving comparables & running Investor Calibration Agent (RAG)...")
    industry = fit.get("industry_guess", "general")
    stage = fit.get("stage_guess", "seed")
    calibration = calibration_agent.run(slides, deck_text_for_agents, model, host, industry=industry, stage=stage)

    # Compute the deterministic score and its readiness band BEFORE the
    # committee runs, so the committee's qualitative verdict is required to
    # be consistent with the number — not produced independently of it.
    overall_score_10 = compute_overall_score(
        clarity.get("score", 0), narrative.get("score", 0), fit.get("fit_score", 0)
    )
    readiness_band = score_to_readiness_band(overall_score_10)

    tick("Running Investor Committee Agent...")
    committee = committee_agent.run(
        clarity, narrative, fit, calibration, model, host,
        deterministic_score=overall_score_10, readiness_band=readiness_band,
    )

    return {
        "clarity": clarity,
        "narrative": narrative,
        "problem_solution": fit,
        "calibration": calibration,
        "committee": committee,
        "overall_score_10": overall_score_10,
        "overall_score_100": round(overall_score_10 * 10, 1),
        "readiness_band": readiness_band,
        "weights": WEIGHTS,
    }