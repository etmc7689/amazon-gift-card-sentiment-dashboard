#!/usr/bin/env python3
"""
score_reviews.py — Balanced 3-class LLM scoring (Amazon Gift Card reviews)
==========================================================================
Loads reviews, draws a BALANCED sample (equal counts of positive / neutral /
negative), and for each review calls an OpenAI-compatible endpoint with the
prompt in prompt.txt to get a predicted SENTIMENT (3-class) and a dominant
EMOTION (one of 8). Writes one JSON object per review.

Ground truth from the review's star rating:
    rating >= 4  ->  "positive"
    rating == 3  ->  "neutral"
    rating <= 2  ->  "negative"

Why balanced? The raw data is ~88% 5-star, so a random sample is dominated by
one class and a trivial "always positive" model looks ~90% accurate. Sampling
equal amounts of each class makes the metric meaningful (chance = 33%).

Stdlib only. Usage:
    python3 score_reviews.py --data <Gift_Cards.jsonl> \
        --output results/balanced_run.jsonl --per-class 30
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BASE_URL = "http://dobolyi.com:9000/v1"
DEFAULT_API_KEY   = "6418"
DEFAULT_MODEL     = "DeepSeek-V4-Flash-0731"

SYSTEM_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "prompt.txt")).read().split("USER PROMPT")[0].strip()
# ^ the system prompt is everything before the "USER PROMPT" section.

VALID_EMOTIONS = {"anger", "anticipation", "disgust", "fear", "joy",
                  "sadness", "surprise", "trust"}

# Constrain the model's output to valid values via vLLM guided JSON.
GUIDED_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "sentiment": {"type": "string",
                      "enum": ["positive", "neutral", "negative"]},
        "emotion": {"type": "string",
                    "enum": sorted(VALID_EMOTIONS)},
    },
    "required": ["sentiment", "emotion"],
    "additionalProperties": False,
})


def label_from_rating(rating):
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    if rating <= 2:
        return "negative"
    return None


def load_balanced_sample(path, per_class, seed=42):
    buckets = {"positive": [], "neutral": [], "negative": []}
    rating_dist = {}
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            rating = rec.get("rating")
            rating_dist[rating] = rating_dist.get(rating, 0) + 1
            lab = label_from_rating(rating)
            if lab is None:
                continue
            if not (rec.get("title") or rec.get("text")):
                continue
            buckets[lab].append({
                "asin": rec.get("asin"),
                "rating": rating,
                "title": rec.get("title") or "",
                "text": rec.get("text") or "",
                "label": lab,
            })
    rng = random.Random(seed)
    sample = []
    summary = {}
    for lab in ("positive", "neutral", "negative"):
        k = min(per_class, len(buckets[lab]))
        sample += rng.sample(buckets[lab], k)
        summary[f"{lab}_sampled"] = k
        summary[f"{lab}_available"] = len(buckets[lab])
    summary["total_rows"] = total
    summary["rating_distribution"] = rating_dist
    return sample, summary


def build_payload(model, user_content):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,          # room for reasoning + JSON
        "guided_json": GUIDED_SCHEMA,  # vLLM: force valid sentiment/emotion values
    }


def call_llm(base_url, api_key, model, user_content, timeout=90):
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(build_payload(model, user_content)).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    msg = body["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        content = msg.get("reasoning") or ""
    return content


def parse_json_loosely(text):
    """Extract {"sentiment":.., "emotion":..} from whatever the model returned."""
    if not text:
        return None
    # 1) whole-string / bracketed JSON
    for cand in (text.strip(), text[text.find("{"):] if "{" in text else ""):
        try:
            obj = json.loads(cand.rstrip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 2) regex field scan
    m = {"sentiment": re.search(r'"sentiment"\s*:\s*"?([a-zA-Z]+)"?', text),
         "emotion":   re.search(r'"emotion"\s*:\s*"?([a-zA-Z]+)"?', text)}
    obj = {}
    for k, mm in m.items():
        if mm:
            obj[k] = mm.group(1).strip().lower()
    return obj or None


def normalize(obj):
    sent = None
    emot = None
    if isinstance(obj, dict):
        sent = obj.get("sentiment")
        emot = obj.get("emotion")
    # accept bare word answers like "positive" or "joy"
    if isinstance(obj, str):
        t = obj.strip().lower()
        if t in ("positive", "neutral", "negative"):
            sent = t
        elif t in VALID_EMOTIONS:
            emot = t
    if isinstance(sent, str):
        sent = sent.lower()
    if isinstance(emot, str):
        emot = emot.lower()
    if sent not in ("positive", "neutral", "negative"):
        sent = None
    if emot not in VALID_EMOTIONS:
        emot = None
    return sent, emot


def classify_one(item, base_url, api_key, model, retries=2):
    user = f"Title: {item['title']}\nText: {item['text']}"
    for attempt in range(retries + 1):
        try:
            raw = call_llm(base_url, api_key, model, user)
            sent, emot = normalize(parse_json_loosely(raw))
            return {"pred_sentiment": sent, "pred_emotion": emot, "raw": raw[:400]}
        except Exception as e:
            if attempt == retries:
                return {"pred_sentiment": None, "pred_emotion": None,
                        "raw": f"ERROR: {e}"}
            time.sleep(1.5 * (attempt + 1))


def evaluate(preds):
    classes = ("positive", "neutral", "negative")
    cm = {a: {b: 0 for b in classes} for a in classes}
    unknown = 0
    for p in preds:
        pred = p["pred_sentiment"]
        if pred not in classes:
            unknown += 1
            continue
        cm[p["label"]][pred] += 1
    total = sum(cm[a][b] for a in classes for b in classes)
    acc = total and sum(cm[c][c] for c in classes) / total
    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[a][c] for a in classes if a != c)
        fn = sum(cm[c][b] for b in classes if b != c)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec,
                        "recall": rec, "f1": f1}
    return {"confusion": cm, "accuracy": acc, "considered": total,
            "unknown": unknown, "per_class": per_class}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="results/balanced_run.jsonl")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    print("Loading balanced sample ...")
    sample, summary = load_balanced_sample(args.data, args.per_class, seed=args.seed)
    print("Sampling summary:", json.dumps(summary))
    print(f"Sampled {len(sample)} reviews, model={args.model}, workers={args.workers}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(classify_one, it, args.base_url, args.api_key, args.model): it
                for it in sample}
        for fut in as_completed(futs):
            item = futs[fut]
            out = fut.result()
            out.update({"asin": item["asin"], "rating": item["rating"],
                        "label": item["label"], "title": item["title"],
                        "text": item["text"]})
            results.append(out)
    print(f"Done in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            r["sample_summary"] = summary
            f.write(json.dumps(r) + "\n")
    print(f"Predictions -> {args.output}")

    metrics, total = evaluate(results)
    print("\n===== 3-CLASS EVALUATION (balanced) =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}  (considered={total}, unknown={metrics['unknown']})")
    print("Confusion matrix (rows=actual, cols=predicted):")
    cols = ("positive", "neutral", "negative")
    head = "actual/pred"
    print(f"{head:>12}" + "".join(f"{c:>10}" for c in cols))
    for a in cols:
        print(f"{a:>12}" + "".join(f"{metrics['confusion'][a][b]:>10}" for b in cols))
    print("\nPer-class:")
    for c in cols:
        pc = metrics["per_class"][c]
        print(f"  {c:>9}: P={pc['precision'] if pc['precision'] is not None else 'n/a'} "
              f"R={pc['recall'] if pc['recall'] is not None else 'n/a'} "
              f"F1={pc['f1'] if pc['f1'] is not None else 'n/a'} "
              f"(tp={pc['tp']}, fp={pc['fp']}, fn={pc['fn']})")


if __name__ == "__main__":
    main()
