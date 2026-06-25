"""
rag/retrieval.py
Real retrieval (TF-IDF + cosine similarity) over the benchmark knowledge
base in benchmark_data.py. No internet, no embedding API, no GPU needed —
scikit-learn's TfidfVectorizer runs instantly on a corpus this size, which
is the right tool for ~30 short reference documents.

This is what makes calibration "RAG" instead of "static text pasted into
every prompt": the snippets returned depend on the ACTUAL deck content
being evaluated (the problem slide text, the solution slide text, etc.),
so two different decks get two different, relevant sets of comparables.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.benchmark_data import BENCHMARKS

_CORPUS = [b["text"] for b in BENCHMARKS]
_VECTORIZER = TfidfVectorizer(stop_words="english")
_MATRIX = _VECTORIZER.fit_transform(_CORPUS)


def retrieve(query: str, dimension: str = None, top_k: int = 3, industry: str = None, stage: str = None):
    """
    Returns the top_k most relevant benchmark snippets for `query`.

    - dimension: if given, restricts the candidate pool to that dimension
      first (e.g. "problem_framing"), then ranks by similarity within it.
      This matters because a query like "we charge $0.80/box" is about
      business model, and we don't want it accidentally matching a "team"
      snippet just because of incidental word overlap.
    - industry: if given, snippets tagged with that industry (or "general")
      are mildly boosted, so e.g. a hardware deck's solution slide retrieves
      the hardware-specific solution_clarity anchor over a generic one when
      both are otherwise close.
    - stage: if given (pre-seed / seed / series-a / series-b), snippets
      tagged for that stage (or "all", meaning stage-agnostic) are
      STRONGLY preferred for stage-sensitive dimensions (traction,
      market_sizing, ask, business_model). This is a hard filter, not a
      soft boost, for those dimensions: a Series-B deck should never be
      compared against a pre-seed-only traction anchor, because the
      expectations are genuinely different, not just stylistically
      different. For stage-agnostic dimensions (problem_framing,
      solution_clarity, team, narrative_flow, clarity_writing), stage has
      no effect since quality bar there doesn't change with funding stage.
    """
    if not query or not query.strip():
        return []

    STAGE_SENSITIVE_DIMENSIONS = {"traction", "market_sizing", "ask", "business_model"}

    candidate_idx = list(range(len(BENCHMARKS)))
    if dimension:
        candidate_idx = [i for i in candidate_idx if BENCHMARKS[i]["dimension"] == dimension]
        if not candidate_idx:
            candidate_idx = list(range(len(BENCHMARKS)))  # fallback: search everything

    if stage and dimension in STAGE_SENSITIVE_DIMENSIONS:
        stage_filtered = [
            i for i in candidate_idx
            if "all" in BENCHMARKS[i]["stages"] or stage in BENCHMARKS[i]["stages"]
        ]
        if stage_filtered:  # only apply if it doesn't empty out the pool entirely
            candidate_idx = stage_filtered

    query_vec = _VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, _MATRIX[candidate_idx]).flatten()

    scored = []
    for local_i, global_i in enumerate(candidate_idx):
        score = float(sims[local_i])
        if industry and BENCHMARKS[global_i]["industry"] in (industry, "general"):
            score += 0.05  # small nudge, doesn't override genuine semantic relevance
        scored.append((score, global_i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return [{**BENCHMARKS[i], "similarity": round(score, 3)} for score, i in top]


def retrieve_for_deck(sections: dict, industry: str = None, stage: str = None, top_k: int = 2):
    """
    sections: dict mapping dimension name -> text to use as the retrieval
    query for that dimension, e.g.:
        {
          "problem_framing": "<text of the Problem slide>",
          "solution_clarity": "<text of the Solution slide>",
          "market_sizing": "<text of the Market slide>",
          "traction": "<text of the Traction slide>",
          "business_model": "<text of the Business Model slide>",
          "team": "<text of the Team slide>",
          "ask": "<text of the Ask slide>",
          "narrative_flow": "<full deck text or a summary>",
          "clarity_writing": "<full deck text>",
        }
    Returns: dict mapping dimension -> list of retrieved snippet dicts.
    Skips dimensions whose query text is empty (e.g. deck has no Ask slide
    at all — that absence itself is meaningful and is reported separately
    by the agents, not papered over here).
    """
    results = {}
    for dim, query_text in sections.items():
        if query_text and query_text.strip():
            results[dim] = retrieve(query_text, dimension=dim, top_k=top_k, industry=industry, stage=stage)
        else:
            results[dim] = []
    return results