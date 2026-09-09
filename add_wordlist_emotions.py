#!/usr/bin/env python3
"""
add_wordlist_emotions.py — NRC word-list emotion annotation
============================================================
Takes a scored run (JSONL produced by score_reviews.py) and adds a
model-free EMOTION prediction for every review using the NRC Emotion
Lexicon (v0.92). No model calls.

Method (the "bag of words / lexical scoring" approach):
  1. Tokenize each review's TITLE + TEXT (lower-case, split on
     non-alphanumerics).
  2. For each token, look up which of the 8 NRC emotion categories it
     is associated with (lexicon gives word <tab> emotion <tab> 0/1).
  3. Sum the per-category scores across all tokens of the review.
  4. PRIMARY emotion = the category with the highest score. Ties are
     broken by a fixed canonical order (anger, anticipation, disgust,
     fear, joy, sadness, surprise, trust) so results are reproducible.

The point is to compare the LLM's contextual emotion reading against a
transparent, purely lexical assignment and see where they diverge
(word weighting ignores context, sarcasm, and negation).

Usage:
    python3 add_wordlist_emotions.py \
        --run results/balanced_run.jsonl \
        --lexicon data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt \
        --out results/balanced_run_emotions.jsonl
"""

import argparse
import json
import os
import re

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
TOKEN_RE = re.compile(r"[a-zA-Z']+")


def load_lexicon(path):
    """Return {word: set(emotions)} for flagged English emotion entries."""
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, cat, flag = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if flag != "1" or cat not in EMOTIONS:
                continue
            lex.setdefault(word.lower(), set()).add(cat)
    return lex


def score_review(title, text, lex):
    tokens = TOKEN_RE.findall((title or "").lower() + " " + (text or "").lower())
    scores = {e: 0 for e in EMOTIONS}
    matched = 0
    for tok in tokens:
        emo = lex.get(tok)
        if emo:
            matched += 1
            for e in emo:
                scores[e] += 1
    # Primary emotion: highest score, ties broken by canonical order.
    # If no word matched the lexicon, there is no supportable emotion.
    primary = None if matched == 0 else max(
        EMOTIONS, key=lambda e: (scores[e], -EMOTIONS.index(e)))
    return {
        "token_count": len(tokens),
        "matched_words": matched,
        "emotion_scores": scores,
        "wordlist_emotion": primary,
        "wordlist_has_match": matched > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Scored run JSONL")
    ap.add_argument("--lexicon", required=True, help="NRC lexicon .txt")
    ap.add_argument("--out", default="results/balanced_run_emotions.jsonl")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    print(f"Lexicon loaded: {len(lex)} words across {EMOTIONS}")

    out_rows = []
    llm_emotion = {}
    wl_emotion = {}
    agree = agree_on_match = total_match = 0
    with open(args.run, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            extra = score_review(rec["title"], rec["text"], lex)
            rec.update(extra)

            llm = rec.get("pred_emotion")
            wl = rec["wordlist_emotion"]
            llm_emotion[llm or "None"] = llm_emotion.get(llm or "None", 0) + 1
            wl_emotion[wl] = wl_emotion.get(wl, 0) + 1
            if llm == wl:
                agree += 1
            if rec["wordlist_has_match"]:
                total_match += 1
                if llm == wl:
                    agree_on_match += 1
            out_rows.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} annotated reviews -> {args.out}\n")

    def show(dist):
        return ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
    print("LLM emotion distribution   :", show(llm_emotion))
    print("Word-list emotion distribu.:", show(wl_emotion))
    n = len(out_rows)
    print(f"\nAgreement (LLM == wordlist): {agree}/{n} = {agree/n:.1%}")
    if total_match:
        print(f"Agreement on reviews w/ a word-list match: {agree_on_match}/{total_match} = {agree_on_match/total_match:.1%}")


if __name__ == "__main__":
    main()
