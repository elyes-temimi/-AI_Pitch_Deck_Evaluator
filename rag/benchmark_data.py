"""
rag/benchmark_data.py
A curated knowledge base of investor-calibration anchors, used by
rag/retrieval.py to find the comparables MOST RELEVANT to the specific deck
being evaluated (not a fixed block injected into every prompt regardless of
content).

Each entry is a paraphrased, general OBSERVABLE CHARACTERISTIC commonly seen
in that tier of startup pitching (drawn from widely known, publicly
discussed patterns in venture pitching commentary — not reproductions of
any single company's actual deck content or copyrighted text). This keeps
the knowledge base legally clean while still being concrete enough to be
useful as a comparison anchor.

Each snippet has:
  - id: stable identifier (so agent output can cite exactly which anchor it used)
  - dimension: which evaluation dimension this anchor is most useful for
               (problem_framing, solution_clarity, market_sizing, traction,
               team, ask, narrative_flow, clarity_writing)
  - tier: exceptional | average | weak
  - industry: general | saas | hardware | marketplace | fintech | healthtech
              (lets retrieval slightly favor industry-relevant anchors when
              the deck's content matches, without needing a separate model
              per industry)
  - text: the actual characteristic description used for retrieval + shown
          to the LLM as the comparable
"""

BENCHMARKS = [
    # --- Problem framing ---
    {"id": "PF01", "dimension": "problem_framing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The problem is stated with a specific, sourced number for frequency or cost (e.g. 'X% of users experience Y every week, costing $Z'), not a vague claim of pain."},
    {"id": "PF02", "dimension": "problem_framing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The problem is framed around a single, sharply defined user segment rather than 'everyone', making the pain easy to picture concretely."},
    {"id": "PF03", "dimension": "problem_framing", "tier": "exceptional", "industry": "fintech", "stages": ["all"],
     "text": "Strong fintech problem slides quantify the cost of friction in money movement or compliance in dollars or hours lost per transaction/case, not just 'banking is broken'."},
    {"id": "PF04", "dimension": "problem_framing", "tier": "exceptional", "industry": "healthtech", "stages": ["all"],
     "text": "Strong healthtech problem slides tie the pain to a measurable clinical or operational outcome (readmission rate, time-to-diagnosis, staff hours), not just 'healthcare is inefficient'."},
    {"id": "PF05", "dimension": "problem_framing", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "The problem is named but supported only by a generic industry statistic that doesn't directly tie to the startup's specific user or use case."},
    {"id": "PF06", "dimension": "problem_framing", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "The problem is asserted as a general trend or industry buzzword ('X industry is broken/outdated') with no evidence of frequency, severity, or who specifically feels it."},
    {"id": "PF07", "dimension": "problem_framing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The problem is stated as a short, universally self-evident inconvenience or cost that any reader immediately recognizes from their own experience (e.g. 'hotels are expensive', 'waiting in line is annoying') — a sourced statistic is not required when the pain is this obviously relatable; demanding a number here would be a false negative."},
    {"id": "PF08", "dimension": "problem_framing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Existing informal or workaround behavior (e.g. people already solving this problem manually through forums, classifieds, or improvised tools) is cited as evidence the pain is real, serving the same validating function as a statistic."},

    # --- Solution clarity ---
    {"id": "SC01", "dimension": "solution_clarity", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The solution sentence directly answers the mechanism of the problem sentence — a reader can trace exactly how the product removes the stated cause of pain, not just that it's 'better' or 'AI-powered'."},
    {"id": "SC02", "dimension": "solution_clarity", "tier": "exceptional", "industry": "hardware", "stages": ["all"],
     "text": "Strong hardware/IoT solutions describe a concrete install/integration path (time, cost, what it replaces) so an operator can picture adoption friction, not just the sensor's capability."},
    {"id": "SC03", "dimension": "solution_clarity", "tier": "exceptional", "industry": "marketplace", "stages": ["all"],
     "text": "Strong marketplace solutions name which side (supply or demand) is harder to acquire and how the product solves that specific cold-start problem, not just 'we connect buyers and sellers'."},
    {"id": "SC04", "dimension": "solution_clarity", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "The solution is described mainly in feature terms (a list of what the product does) without explicitly closing the loop back to the stated problem."},
    {"id": "SC05", "dimension": "solution_clarity", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "The solution is described in marketing adjectives ('seamless', 'revolutionary', 'next-generation') without explaining the underlying mechanism at all."},
    {"id": "SC06", "dimension": "solution_clarity", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The solution is structured as a point-for-point mirror of the problem statement — each distinct pain point named in the problem has a directly corresponding clause in the solution (e.g. problem: 'expensive, disconnected, hard to book' / solution: 'cheaper, shared culture, easy booking'). This is one of the clearest possible forms of causal alignment, even when expressed in very short, simple sentences — brevity here is a strength, not a sign of missing detail."},

    # --- Market sizing ---
    {"id": "MS01", "dimension": "market_sizing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Market size is derived bottom-up: number of target customers x price/unit x frequency, shown as a calculation, not just a cited '$X billion TAM' figure from a research report."},
    {"id": "MS02", "dimension": "market_sizing", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "Market size cites a large top-down industry figure (e.g. 'the global X market is $Y billion') without narrowing to the serviceable segment the startup can actually reach."},
    {"id": "MS03", "dimension": "market_sizing", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "Market size is a single huge number with no derivation at all, often from an unrelated adjacent industry, used to imply opportunity without rigor."},
    {"id": "MS04", "dimension": "market_sizing", "tier": "exceptional", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "At pre-seed/seed, a bottom-up TAM/SAM estimate is sufficient evidence of opportunity on its own, even without traction data to corroborate it yet — the calculation method matters more than proof the company has captured any of it."},
    {"id": "MS05", "dimension": "market_sizing", "tier": "exceptional", "industry": "general", "stages": ["series-a", "series-b"],
     "text": "At Series A or later, market sizing should be corroborated by the company's own revenue run-rate or market share captured so far, not just a theoretical TAM calculation — investors expect the size claim to connect to demonstrated traction, not stand alone."},

    # --- Traction ---
    {"id": "TR01", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Traction is shown as a trend over multiple time periods (week-over-week or month-over-month), demonstrating momentum, not a single static snapshot number."},
    {"id": "TR02", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Traction includes a revenue or retention signal (conversion to paid, renewal, repeat usage), not only top-of-funnel signups."},
    {"id": "TR03", "dimension": "traction", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "Traction is a single vanity metric (e.g. total signups or downloads) with no context on retention, conversion, or trend direction."},
    {"id": "TR04", "dimension": "traction", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "Traction is absent, or substituted with a statement of future intent ('we expect to reach X users by Y') rather than any actual evidence to date."},
    {"id": "TR05", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "Pre-launch market validation is shown via existing informal/adjacent-platform behavior (e.g. people already organizing this activity through forums, classifieds, or general-purpose platforms not built for it) — at pre-seed/seed stage, before the product itself exists, this kind of behavioral evidence is a strong substitute for product usage metrics, not a weaker form of traction."},
    {"id": "TR06", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "At pre-seed/seed, strong traction looks like: a working prototype with real early users, a waitlist with genuine signups, or a small paid pilot — revenue scale is not expected yet; what matters is evidence that real people engaged, not a polished growth curve."},
    {"id": "TR07", "dimension": "traction", "tier": "weak", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "At pre-seed/seed, demanding large-scale revenue or growth-curve proof before there's even a product is an unreasonable bar — but a deck with NO evidence at all (not even informal validation or a waitlist) is still weak for this stage."},
    {"id": "TR08", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["series-a"],
     "text": "At Series A, strong traction requires demonstrated product-market fit: consistent month-over-month revenue growth, retention/cohort data showing usage holds up over time, and at least one repeatable, scalable acquisition channel — informal pre-launch signals are no longer sufficient at this stage."},
    {"id": "TR09", "dimension": "traction", "tier": "weak", "industry": "general", "stages": ["series-a", "series-b"],
     "text": "Citing only early-stage validation signals (informal pre-launch demand, a small unpaid pilot, or a waitlist) as the primary evidence of traction is a weak pattern at Series A or beyond — investors at this stage expect revenue and retention proof, not pre-launch signals."},
    {"id": "TR10", "dimension": "traction", "tier": "exceptional", "industry": "general", "stages": ["series-b"],
     "text": "At Series B, strong traction requires proven, scalable unit economics (positive or clearly improving contribution margin), multiple acquisition channels working simultaneously, and evidence the model works beyond the original beachhead market or segment."},

    # --- Business model (stage-sensitive: how much mechanism/proof is expected) ---
    {"id": "BM01", "dimension": "business_model", "tier": "exceptional", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "At pre-seed/seed, a clearly stated revenue mechanism (e.g. a specific commission rate, subscription price, or per-unit fee) is sufficient, even without proven unit economics yet — clarity of the mechanism matters more than proof it scales profitably."},
    {"id": "BM02", "dimension": "business_model", "tier": "weak", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "No revenue mechanism is stated at all, or monetization is deferred with 'we'll figure it out once we have users' — even at the earliest stage, investors expect at least a hypothesis for how money will be made."},
    {"id": "BM03", "dimension": "business_model", "tier": "exceptional", "industry": "general", "stages": ["series-a", "series-b"],
     "text": "At Series A or later, the business model section should show real unit economics in practice (actual achieved margin, CAC vs LTV, or payback period from real data), not just a stated pricing mechanism — investors expect the model to be validated, not merely described."},

    # --- Team ---
    {"id": "TM01", "dimension": "team", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Team slide shows specific, relevant prior outcomes directly tied to this problem space (e.g. 'built and scaled X at Y company to Z outcome'), not just job titles."},
    {"id": "TM02", "dimension": "team", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "Team slide lists credentials (where they worked, studied) without connecting that experience to why they're specifically positioned to solve this problem."},
    {"id": "TM03", "dimension": "team", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "Team slide states roles only ('CEO', 'CTO') with generic descriptors like 'passionate' or 'experienced' and no verifiable track record at all."},

    # --- Ask ---
    {"id": "AS01", "dimension": "ask", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The ask is explicit and specific: an amount, what it will be used for, and the concrete milestone it unlocks (e.g. 'X months of runway to reach Y metric')."},
    {"id": "AS02", "dimension": "ask", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "An amount is given but the use of funds and the milestone it's meant to unlock are vague or generic ('to grow the team and scale')."},
    {"id": "AS03", "dimension": "ask", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "There is no explicit ask at all, or the deck ends on a vision statement instead of a concrete funding request."},
    {"id": "AS04", "dimension": "ask", "tier": "exceptional", "industry": "general", "stages": ["pre-seed", "seed"],
     "text": "At pre-seed/seed, a typical, credible ask is in the hundreds-of-thousands to low-single-digit-millions range, tied to reaching specific product or early-traction milestones — an ask far larger than this for the stage (e.g. tens of millions) would itself be a red flag, not a strength."},
    {"id": "AS05", "dimension": "ask", "tier": "exceptional", "industry": "general", "stages": ["series-a"],
     "text": "At Series A, a typical, credible ask is commonly in the mid-single-digit to ~$15M range, tied to scaling a proven acquisition channel or expanding to new markets/segments, building on demonstrated product-market fit."},
    {"id": "AS06", "dimension": "ask", "tier": "exceptional", "industry": "general", "stages": ["series-b"],
     "text": "At Series B, the ask is typically tied to scaling operations and improving unit economics across multiple markets, usually a larger round focused on growth efficiency rather than initial proof of concept."},

    # --- Narrative flow ---
    {"id": "NF01", "dimension": "narrative_flow", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "Each slide is the logical consequence of the previous one — the deck reads as one continuous argument building to the ask, not a checklist of independent topics."},
    {"id": "NF02", "dimension": "narrative_flow", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "The deck follows the expected investor arc (Problem -> Solution -> Product -> Market -> Business Model -> Traction -> GTM -> Team -> Ask) without skipping a step an investor would expect to see before deciding."},
    {"id": "NF03", "dimension": "narrative_flow", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "Most expected sections are present but feel like a checklist — each slide is self-contained rather than explicitly building on the one before it."},
    {"id": "NF04", "dimension": "narrative_flow", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "Key sections (commonly: a clearly stated problem, traction, or the ask) are missing entirely, or the deck jumps between topics (e.g. vision, then product, then back to problem) without a clear through-line."},

    # --- Clarity / writing ---
    {"id": "CW01", "dimension": "clarity_writing", "tier": "exceptional", "industry": "general", "stages": ["all"],
     "text": "One clear idea per slide; a reader can grasp each slide's point in roughly 10 seconds without re-reading. Very short, plain sentences are a hallmark of strong founder communication — do not mistake brevity or simple language for missing detail or lack of substance."},
    {"id": "CW02", "dimension": "clarity_writing", "tier": "average", "industry": "general", "stages": ["all"],
     "text": "Some slides are dense with multiple ideas or long paragraphs, requiring the reader to slow down and re-read to extract the key point."},
    {"id": "CW03", "dimension": "clarity_writing", "tier": "weak", "industry": "general", "stages": ["all"],
     "text": "Slides are overloaded with text, repeat the same point across multiple slides, or contain grammar/spelling errors that distract from the content."},
]


def all_dimensions():
    return sorted(set(b["dimension"] for b in BENCHMARKS))