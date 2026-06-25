"""
report.py
Renders the 5-agent evaluation result into a readable Markdown report.
"""


def _bullets(items):
    if not items:
        return "_None noted._\n"
    return "\n".join(f"- {x}" for x in items) + "\n"


def render_markdown(result: dict, source_name: str, extraction_warnings=None) -> str:
    clarity = result["clarity"]
    narrative = result["narrative"]
    fit = result["problem_solution"]
    calibration = result["calibration"]
    committee = result["committee"]
    weights = result["weights"]

    md = []
    md.append("# Pitch Deck Evaluation Report\n")
    md.append(f"**Source:** `{source_name}`\n")
    md.append(f"**Overall score:** {result['overall_score_10']}/10  "
               f"({result['overall_score_100']}/100)\n")
    md.append(
        f"_Weighted: Clarity {weights['clarity']*100:.0f}% · "
        f"Narrative {weights['narrative']*100:.0f}% · "
        f"Problem-Solution Fit {weights['fit']*100:.0f}%_\n"
    )
    md.append(f"**Benchmark alignment (RAG calibration, reported separately):** "
               f"{calibration.get('benchmark_alignment_score', 'N/A')}/10\n")
    md.append(f"**Investment readiness:** {committee.get('investment_readiness', 'N/A')}\n")

    md.append("## Investor Committee Verdict\n")
    md.append(f"{committee.get('recommendation', '')}\n")
    md.append("**Key strengths:**")
    md.append(_bullets(committee.get("key_strengths")))
    md.append("**Key weaknesses:**")
    md.append(_bullets(committee.get("key_weaknesses")))
    md.append("**Top risks:**")
    md.append(_bullets(committee.get("top_risks")))
    if committee.get("unable_to_assess"):
        md.append("**Could not assess (not a flaw — info absent from extracted deck):**")
        md.append(_bullets(committee.get("unable_to_assess")))
    if committee.get("band_disagreement_note"):
        md.append(f"**Note:** committee's qualitative read differed from the deterministic "
                   f"score band: {committee['band_disagreement_note']}\n")
    if committee.get("contradictions_resolved"):
        md.append("**Contradictions between analysts (resolved):**")
        md.append(_bullets(committee.get("contradictions_resolved")))

    md.append("---\n")
    md.append("## 1. Investor Communication & Deck Clarity\n")
    md.append(f"**Score:** {clarity.get('score', 'N/A')}/10\n")
    md.append("**Strengths:**")
    md.append(_bullets(clarity.get("strengths")))
    md.append("**Weaknesses:**")
    md.append(_bullets(clarity.get("weaknesses")))
    md.append("**Recommendations:**")
    md.append(_bullets(clarity.get("recommendations")))
    md.append(f"**Notes:** {clarity.get('notes', '')}\n")

    md.append("## 2. Narrative & Founder Storytelling\n")
    md.append(f"**Score:** {narrative.get('score', 'N/A')}/10\n")
    md.append(f"**Missing sections:** {', '.join(narrative.get('missing_sections', [])) or 'None'}")
    md.append(f"**Weak sections:** {', '.join(narrative.get('weak_sections', [])) or 'None'}")
    md.append(f"**Flow consistency:** {narrative.get('flow_consistency', 'N/A')}\n")
    md.append("**Strengths:**")
    md.append(_bullets(narrative.get("strengths")))
    md.append("**Weaknesses:**")
    md.append(_bullets(narrative.get("weaknesses")))
    md.append(f"**Notes:** {narrative.get('notes', '')}\n")

    md.append("## 3. Problem-Solution Fit (Investment Lens)\n")
    md.append(f"**Fit score:** {fit.get('fit_score', 'N/A')}/10\n")
    md.append(f"- **Problem (as interpreted):** {fit.get('problem', 'N/A')}")
    md.append(f"- **Solution (as interpreted):** {fit.get('solution', 'N/A')}")
    md.append(f"- **Causal link quality:** {fit.get('causal_link_quality', 'N/A')}")
    md.append(f"- **Pain severity:** {fit.get('pain_severity', 'N/A')}")
    md.append(f"- **Pain frequency:** {fit.get('pain_frequency', 'N/A')}")
    md.append(f"- **Scalability:** {fit.get('scalability', 'N/A')}")
    md.append(f"- **Industry guess:** {fit.get('industry_guess', 'N/A')}")
    md.append(f"- **Stage guess:** {fit.get('stage_guess', 'N/A')} — {fit.get('stage_reasoning', '')}\n")
    md.append("**Strengths:**")
    md.append(_bullets(fit.get("strengths")))
    md.append("**Risks:**")
    md.append(_bullets(fit.get("risks")))
    md.append(f"**Notes:** {fit.get('notes', '')}\n")

    md.append("## 4. Investor Calibration (RAG — retrieved comparables)\n")
    md.append(f"**Benchmark alignment score:** {calibration.get('benchmark_alignment_score', 'N/A')}/10\n")
    per_dim = calibration.get("per_dimension", {})
    for dim, info in per_dim.items():
        md.append(f"- **{dim}:** {info.get('alignment', 'N/A')} — {info.get('comparison', '')}")
    md.append("")
    md.append("**Standout strengths (vs. exceptional-tier patterns):**")
    md.append(_bullets(calibration.get("standout_strengths")))
    md.append("**Gaps vs. benchmark (vs. average/weak-tier patterns):**")
    md.append(_bullets(calibration.get("gaps_vs_benchmark")))

    trace = calibration.get("_retrieval_trace")
    if trace:
        md.append("<details><summary>Retrieval trace (which reference snippets were retrieved, and why)</summary>\n")
        for dim, items in trace.items():
            if not items:
                md.append(f"- **{dim}**: no reference retrieved\n")
                continue
            for it in items:
                md.append(f"- **{dim}** -> `{it['id']}` (tier: {it['tier']}, similarity: {it['similarity']}): {it['text']}")
        md.append("\n</details>\n")

    if extraction_warnings:
        md.append("## Limitations Noted\n")
        md.append(_bullets(extraction_warnings))

    return "\n".join(md)