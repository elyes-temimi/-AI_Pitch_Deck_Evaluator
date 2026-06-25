# AI Pitch Deck Evaluator — Multi-Agent, RAG-Calibrated MVP

An investor-style pitch deck evaluator built as five specialized LLM agents
running locally on Llama 3.1 via Ollama. No API key, no data leaves your
machine.

## Architecture

```
Deck input (.json / .pptx / .pdf / .txt)
            │
            ▼
extraction/extractor.py — normalizes any input into [{slide, title, content}]
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
Clarity  Narrative  Problem-Solution
 Agent    Agent      Agent  (also guesses industry: saas/hardware/fintech/...)
   │        │              │
   └────────┴──────┬───────┘
                    ▼
      Investor Calibration Agent (RAG)
      1. Pulls THIS deck's actual Problem/Solution/Market/Traction/Team/Ask text
      2. rag/retrieval.py: TF-IDF search over rag/benchmark_data.py
         → retrieves the comparables most relevant to THIS deck's content
      3. LLM compares deck content against retrieved comparables, citing ids
                    │
                    ▼
         Investor Committee Agent
         (synthesizes all 4 analysts' findings into a verdict)
                    │
                    ▼
   Deterministic weighted score (computed in Python, not the LLM)
   overall = clarity*0.25 + narrative*0.30 + fit*0.45
   Benchmark Alignment (RAG) reported SEPARATELY, not blended in
                    │
                    ▼
   report.py → Markdown report  /  app.py → Streamlit UI
```

## What makes this real RAG, not a static prompt block

Two different decks retrieve **different** comparables, because retrieval
runs against each deck's actual slide text:

- A deck whose Problem slide cites a specific cost/frequency number retrieves
  the "exceptional" problem-framing anchor (high similarity).
- A deck whose Team slide uses generic language ("passionate", no track
  record) retrieves the "weak" team anchor — verified on `test_decks/weak_deck.json`,
  where it scored 0.54 similarity against the weak-tier anchor, the highest
  similarity score observed anywhere in testing.
- A deck missing a slide entirely (e.g. no Ask slide) retrieves **nothing**
  for that dimension, and the report says so explicitly rather than forcing
  a comparison.

You can run `python -c "from agents.calibration_agent import gather_retrieval_context; ..."`
live in the interview and show different decks pulling different anchors —
that's the proof this is retrieval, not a fixed block.

## Honest limitation: TF-IDF is lexical, not semantic

`rag/retrieval.py` uses TF-IDF + cosine similarity (scikit-learn), not
embeddings. This was a deliberate choice — no extra heavy dependency, no
GPU, instant on a ~30-document corpus. The tradeoff: it matches on shared
vocabulary, not deeper meaning. In testing, this occasionally ranked a
wrong-tier anchor highest by accidental word overlap (e.g. a strong deck's
specific, track-record-based team slide briefly out-scored by a generic
"weak" anchor). The fix applied: retrieval returns the **top 3** candidates
per dimension instead of just 1, so the LLM sees enough candidates to reason
past TF-IDF's ranking noise rather than being shown only a wrong match.

**Say this limitation out loud in your interview before they find it.** It's
a correct, deliberate scope decision for a 24h MVP (swapping in
sentence-transformers embeddings later is a one-file change, not a
redesign — `retrieval.py` already isolates this concern).

## Why fine-tuning a model was explicitly rejected

A credible fine-tune needs a labeled dataset with verified investor
outcomes, a GPU, and held-out evaluation to prove it isn't overfit to a
handful of examples — none of which fits a 24h window. A rushed fine-tune
on a tiny dataset typically makes judgment *worse* by overfitting to quirks
of those few examples, while llama3.1's pretrained reasoning about venture
logic is already a stronger foundation. Retrieval-augmented calibration
gets the "judges like an investor, grounded in real patterns" claim without
that risk.

## Why the headline score is computed in Python, not by the LLM

Local 8B models are inconsistent at arithmetic. Computing
`overall = clarity*0.25 + narrative*0.30 + fit*0.45` in code keeps the
headline number reproducible and auditable. Problem-Solution Fit gets the
highest weight because the brief explicitly calls it the "core investment
validation layer." Benchmark Alignment (the calibration agent's score) is
reported as a separate number, not folded into the headline score — it
measures similarity to known patterns, which is a different question from
whether the deck itself is good.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1
# Ollama usually runs as a background service automatically. If not: ollama serve
```

## Usage

### Streamlit UI (recommended for the demo)
```bash
streamlit run app.py
```
Three one-click demo buttons: sample deck, strong test deck, weak test deck.
Tab 4 ("Investor Calibration") shows the retrieval trace live — exactly
which comparables were retrieved for that specific deck.

### CLI
```bash
python main.py sample_deck.json
python main.py my_deck.pptx --json result.json
```

## Input formats

- **JSON** (explicitly called out as acceptable by the assignment):
  `[{"slide": 1, "title": "Problem", "content": "..."}]`
- **.pptx** — parsed slide-by-slide (title placeholder, body text, tables, notes)
- **.pdf** — one "slide" per page
- **.txt/.md** — slides separated by a line containing only `---`

## Testing on successful vs. failed patterns

`test_decks/strong_deck.json` and `test_decks/weak_deck.json` are designed
to exercise the full range:
- Strong: bottom-up market sizing, trend-based traction, specific team track
  record, explicit ask — should score high on both the headline score and
  benchmark alignment.
- Weak: no Problem/Solution/Ask slides at all, vague claims, single vanity
  metric — should score low and should visibly retrieve weak-tier anchors.

Run both, compare the reports side by side — that comparison is the
strongest moment in your demo.

## File-by-file

| File | Role |
|---|---|
| `extraction/extractor.py` | Normalizes JSON/PPTX/PDF/TXT into a canonical slide list |
| `llm/ollama_client.py` | Ollama HTTP client + robust JSON parsing/retry |
| `rag/benchmark_data.py` | ~30 tagged, retrievable investor-calibration anchors |
| `rag/retrieval.py` | Real TF-IDF retrieval over the benchmark corpus |
| `agents/clarity_agent.py` | Dimension 1 |
| `agents/narrative_agent.py` | Dimension 2 |
| `agents/fit_agent.py` | Dimension 3 (also guesses industry for retrieval) |
| `agents/calibration_agent.py` | RAG-powered investor calibration (requirement #4) |
| `agents/committee_agent.py` | Synthesizes all 4 into a final verdict |
| `pipeline.py` | Orchestrates the 5-agent run + deterministic scoring |
| `report.py` | Renders the result into Markdown, including the retrieval trace |
| `app.py` | Streamlit UI |
| `main.py` | CLI entrypoint |
| `test_decks/` | Strong and weak example decks for the demo |
