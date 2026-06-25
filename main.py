#!/usr/bin/env python3
"""
main.py — AI Pitch Deck Evaluator, CLI mode (5-agent, RAG-calibrated).

Usage:
    python main.py deck.json
    python main.py deck.pptx
    python main.py deck.pdf
    python main.py deck.txt --model llama3.1 --host http://localhost:11434
    python main.py deck.pptx --out report.md --json result.json
"""

import argparse
import json
import sys
from pathlib import Path

from extraction.extractor import extract_deck, deck_to_prompt_text
from pipeline import evaluate_deck
from report import render_markdown


def main():
    parser = argparse.ArgumentParser(description="AI Pitch Deck Evaluator (5-agent, RAG-calibrated)")
    parser.add_argument("deck_path", help="Path to the pitch deck (.json, .pptx, .pdf, or .txt)")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    deck_path = Path(args.deck_path)
    out_path = Path(args.out) if args.out else deck_path.with_name(deck_path.stem + "_report.md")

    print(f"[1/2] Reading {deck_path.name} ...")
    try:
        slides, warnings = extract_deck(str(deck_path))
    except Exception as e:
        print(f"ERROR reading deck: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"      -> {len(slides)} slide(s) parsed.")
    for w in warnings:
        print(f"      WARNING: {w}")

    deck_text = deck_to_prompt_text(slides)

    print(f"[2/2] Running 5-agent evaluation with '{args.model}' via {args.host} ...")

    def progress(msg):
        print(f"      {msg}")

    try:
        result = evaluate_deck(slides, deck_text, model=args.model, host=args.host, progress_cb=progress,
                                extraction_warnings=warnings)
    except Exception as e:
        print(f"ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    md = render_markdown(result, source_name=str(deck_path), extraction_warnings=warnings)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReport written -> {out_path}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Raw JSON written -> {args.json}")

    print(f"\nOverall score: {result['overall_score_10']}/10 ({result['overall_score_100']}/100)")
    print(f"Benchmark alignment (RAG): {result['calibration'].get('benchmark_alignment_score')}/10")
    print(f"Investment readiness: {result['committee'].get('investment_readiness')}")


if __name__ == "__main__":
    main()