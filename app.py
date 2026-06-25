"""
app.py — Streamlit UI for the AI Pitch Deck Evaluator (5-agent, RAG-calibrated).

Run with:
    streamlit run app.py
"""

import json
import time
import streamlit as st

from extraction.extractor import extract_deck, deck_to_prompt_text
from pipeline import evaluate_deck, WEIGHTS
from report import render_markdown
from llm.ollama_client import OllamaError

st.set_page_config(page_title="AI Pitch Deck Evaluator", page_icon="📊", layout="wide")

st.title("📊 AI Pitch Deck Evaluator")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value="llama3.1")
    host = st.text_input("Ollama host", value="http://localhost:11434")
    st.markdown("---")
    st.markdown("**Scoring weights**")
    st.write(f"Clarity: {WEIGHTS['clarity']*100:.0f}%")
    st.write(f"Narrative: {WEIGHTS['narrative']*100:.0f}%")
    st.write(f"Problem-Solution Fit: {WEIGHTS['fit']*100:.0f}%")
    st.markdown("---")
    st.markdown(
        "**Architecture**\n\n"
        "1. Clarity Agent\n"
        "2. Narrative Agent\n"
        "3. Problem-Solution Agent\n"
        "4. Investor Committee Agent (synthesizes 1-3)\n\n"
        "Headline score is computed from the three agent scores, not by the LLM."
    )

uploaded = st.file_uploader(
    "Upload a pitch deck",
    type=["json", "pptx", "pdf", "txt"],
    help="JSON: [{'slide':1,'title':'Problem','content':'...'}] · or a real deck file.",
)

manual_text = st.text_area(
    "Or paste your deck text here",
    help="Paste title and content text, or a short pitch deck narrative.",
    height=220,
)

run_clicked = st.button(
    "Run evaluation", type="primary",
    disabled=not (uploaded or manual_text.strip()),
)


def get_slides_and_warnings():
    if uploaded is not None:
        file_bytes = uploaded.read()
        return extract_deck(file_bytes, filename=uploaded.name)
    return [{"slide": 1, "title": "Manual input", "content": manual_text}], []


if run_clicked:
    try:
        slides, warnings = get_slides_and_warnings()
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")
        st.stop()

    for w in warnings:
        st.warning(w)

    with st.expander(f"Parsed {len(slides)} slide(s) — click to preview"):
        for s in slides:
            st.markdown(f"**Slide {s['slide']}: {s['title']}**")
            st.text(s["content"][:500] if s["content"] else "(empty)")

    deck_text = deck_to_prompt_text(slides)

    status = st.status("Running multi-agent evaluation...", expanded=True)

    def progress(msg):
        status.write(msg)

    try:
        start = time.time()
        result = evaluate_deck(slides, deck_text, model=model, host=host, progress_cb=progress)
        elapsed = time.time() - start
        status.update(label=f"Evaluation complete in {elapsed:.0f}s", state="complete")
    except OllamaError as e:
        status.update(label="Evaluation failed", state="error")
        st.error(str(e))
        st.stop()
    except Exception as e:
        status.update(label="Evaluation failed", state="error")
        st.error(f"Unexpected error: {e}")
        st.stop()

    st.markdown("---")
    c1, c2 ,c3 = st.columns(3)
    c1.metric("Overall score", f"{result['overall_score_10']}/10")
    c2.metric("Benchmark alignment (RAG)", f"{result['calibration'].get('benchmark_alignment_score', 'N/A')}/10")
    c3.metric("Investment readiness", result["committee"].get("investment_readiness", "N/A"))
    

    st.subheader("🧑‍💼 Investor Committee Verdict")
    st.write(result["committee"].get("recommendation", ""))
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("**Key strengths**")
        for x in result["committee"].get("key_strengths", []):
            st.markdown(f"- {x}")
    with cc2:
        st.markdown("**Key weaknesses**")
        for x in result["committee"].get("key_weaknesses", []):
            st.markdown(f"- {x}")
    with cc3:
        st.markdown("**Top risks**")
        for x in result["committee"].get("top_risks", []):
            st.markdown(f"- {x}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Clarity & Communication", "2. Narrative & Storytelling",
        "3. Problem-Solution Fit", "4. Investor Calibration (RAG)",
    ])

    with tab1:
        clarity = result["clarity"]
        st.metric("Clarity score", f"{clarity.get('score', 'N/A')}/10")
        st.markdown("**Strengths**")
        for x in clarity.get("strengths", []):
            st.markdown(f"- {x}")
        st.markdown("**Weaknesses**")
        for x in clarity.get("weaknesses", []):
            st.markdown(f"- {x}")
        st.markdown("**Recommendations**")
        for x in clarity.get("recommendations", []):
            st.markdown(f"- {x}")
        st.caption(clarity.get("notes", ""))

    with tab2:
        narrative = result["narrative"]
        st.metric("Narrative score", f"{narrative.get('score', 'N/A')}/10")
        st.markdown(f"**Missing sections:** {', '.join(narrative.get('missing_sections', [])) or 'None'}")
        st.markdown(f"**Weak sections:** {', '.join(narrative.get('weak_sections', [])) or 'None'}")
        st.markdown(f"**Flow consistency:** {narrative.get('flow_consistency', '')}")
        st.markdown("**Strengths**")
        for x in narrative.get("strengths", []):
            st.markdown(f"- {x}")
        st.markdown("**Weaknesses**")
        for x in narrative.get("weaknesses", []):
            st.markdown(f"- {x}")
        st.caption(narrative.get("notes", ""))

    with tab3:
        fit = result["problem_solution"]
        st.metric("Problem-Solution fit score", f"{fit.get('fit_score', 'N/A')}/10")
        st.markdown(f"**Problem (interpreted):** {fit.get('problem', '')}")
        st.markdown(f"**Solution (interpreted):** {fit.get('solution', '')}")
        st.markdown(f"**Causal link quality:** {fit.get('causal_link_quality', '')}")
        st.markdown(f"**Industry guess:** {fit.get('industry_guess', '')}")
        st.markdown(f"**Pain severity:** {fit.get('pain_severity', '')} · "
                     f"**Pain frequency:** {fit.get('pain_frequency', '')} · "
                     f"**Scalability:** {fit.get('scalability', '')}")
        st.markdown("**Strengths**")
        for x in fit.get("strengths", []):
            st.markdown(f"- {x}")
        st.markdown("**Risks**")
        for x in fit.get("risks", []):
            st.markdown(f"- {x}")
        st.caption(fit.get("notes", ""))

    with tab4:
        calibration = result["calibration"]
        st.metric("Benchmark alignment", f"{calibration.get('benchmark_alignment_score', 'N/A')}/10")
        st.markdown("**Per-dimension comparison vs. retrieved comparables:**")
        for dim, info in calibration.get("per_dimension", {}).items():
            st.markdown(f"- **{dim}**: {info.get('alignment', '')} — {info.get('comparison', '')}")
        st.markdown("**Standout strengths (vs. exceptional-tier patterns)**")
        for x in calibration.get("standout_strengths", []):
            st.markdown(f"- {x}")
        st.markdown("**Gaps vs. benchmark (vs. average/weak-tier patterns)**")
        for x in calibration.get("gaps_vs_benchmark", []):
            st.markdown(f"- {x}")
        with st.expander("🔍 Retrieval trace — exactly what was retrieved for THIS deck, and why"):
            trace = calibration.get("_retrieval_trace", {})
            for dim, items in trace.items():
                st.markdown(f"**{dim}**")
                if not items:
                    st.caption("No reference retrieved — this deck has no slide content for this dimension.")
                    continue
                for it in items:
                    st.markdown(f"- `{it['id']}` (tier: **{it['tier']}**, similarity: {it['similarity']}): {it['text']}")

    st.markdown("---")
    md_report = render_markdown(
        result,
        source_name=(uploaded.name if uploaded else "manual_input"),
        extraction_warnings=warnings,
    )
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇ Download Markdown report", md_report, file_name="pitch_deck_report.md")
    dl2.download_button("⬇ Download raw JSON", json.dumps(result, indent=2), file_name="pitch_deck_result.json")
