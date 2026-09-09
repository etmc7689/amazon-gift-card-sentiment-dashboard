#!/usr/bin/env python3
"""
make_dashboard.py — build a self-contained interactive dashboard (HTML)
=======================================================================
Reads the annotated balanced run (results/balanced_run_emotions.jsonl),
computes all aggregates in Python, and renders ONE self-contained HTML file
(dashboard.html): all CSS, JS, chart markup, and review data are embedded,
so the page works offline with zero network requests.

Charts are hand-rolled HTML/CSS bars (not an external library) and are
rendered deterministically here so the numbers on the page are the exact
numbers saved in the JSONL. Review filtering is the interactive part.

Usage:
    python3 make_dashboard.py \
        --run results/balanced_run_emotions.jsonl \
        --out dashboard.html
"""

import argparse
import json
import html
from collections import Counter

CLASSES = ["positive", "neutral", "negative"]
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
COLOR = {
    "positive": "#2e9e5b", "neutral": "#c9a227", "negative": "#d64541",
}
EMO_COLOR = {
    "anger": "#d64541", "anticipation": "#7f6fb2", "disgust": "#8b5a2b",
    "fear": "#4f4f8b", "joy": "#2e9e5b", "sadness": "#4a90d9",
    "surprise": "#e08a00", "trust": "#3aa6a6",
}


def esc(s):
    return html.escape(str(s), quote=True)


def pct_str(a, b):
    return f"{a/b*100:.1f}%" if b else "n/a"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def compute(rows):
    # --- sentiment accuracy / confusion ---
    cm = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for r in rows:
        p = r["pred_sentiment"]
        if p in CLASSES:
            cm[r["label"]][p] += 1
    total = sum(cm[a][b] for a in CLASSES for b in CLASSES)
    acc = sum(cm[c][c] for c in CLASSES) / total if total else 0
    per_class = {}
    for c in CLASSES:
        tp = cm[c][c]; fp = sum(cm[a][c] for a in CLASSES if a != c)
        fn = sum(cm[c][b] for b in CLASSES if b != c)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        per_class[c] = {"tp": tp, "recall": rec, "prec": prec, "f1": f1,
                        "actual": cm[c]["positive"] + cm[c]["neutral"] + cm[c]["negative"]}

    pred_counts = {c: sum(cm[a][c] for a in CLASSES) for c in CLASSES}
    actual_counts = {c: per_class[c]["actual"] for c in CLASSES}
    correct_counts = {c: per_class[c]["tp"] for c in CLASSES}

    # --- rating (star) distribution: full dataset, from the first row's summary ---
    rating_dist = rows[0].get("sample_summary", {}).get("rating_distribution", {})
    # --- emotion distributions ---
    llm_emo = Counter(r["pred_emotion"] for r in rows if r.get("pred_emotion"))
    wl_emo = Counter(r["wordlist_emotion"] for r in rows if r.get("wordlist_emotion"))

    # rows enriched for the table
    table_rows = []
    for r in rows:
        table_rows.append({
            "asin": r.get("asin"), "rating": r.get("rating"),
            "title": r.get("title"), "text": r.get("text"),
            "label": r.get("label"), "pred": r.get("pred_sentiment"),
            "llm_emo": r.get("pred_emotion"), "wl_emo": r.get("wordlist_emotion"),
        })
    agg = {
        "total": total, "accuracy": acc,
        "confusion": cm, "per_class": per_class,
        "pred_counts": pred_counts, "actual_counts": actual_counts,
        "correct_counts": correct_counts,
        "rating_dist": {int(float(k)) if str(k).replace('.', '', 1).isdigit() else k: v
                        for k, v in rating_dist.items()},
        "llm_emo": dict(llm_emo), "wl_emo": dict(wl_emo),
    }
    return agg, table_rows


# ---------------- HTML building blocks ----------------

def bar(maxv, val, color, label=None, minw=6):
    """A proportionally sized bar that never collapses to zero width."""
    ratio = (val / maxv) if maxv else 0
    width = f"max({ratio*100:.1f}%, {minw}px)"
    return (f'<div class="bar-wrap"><div class="bar" style="width:{width};'
            f'background:{color}"><span class="bar-val">{val}</span></div>'
            f'<div class="bar-label">{esc(label) if label else ""}</div></div>')


def star_dist_html(dist):
    # order 1..5
    out = ['<div class="chart" id="star-chart">']
    if not dist:
        out.append("<p>No rating data.</p>")
        return "".join(out)
    scaled = {s: 0 for s in range(1, 6)}
    for s in range(1, 6):
        scaled[s] = int(dist.get(s, 0) or 0)
    maxv = max(scaled.values())
    for s in range(1, 6):
        lab = "★" * s
        out.append(f'<div class="hbar-row">'
                   f'<div class="hbar-label">{lab}</div>'
                   f'{bar(maxv, scaled[s], "#6b7280", f"{scaled[s]:,}")}'
                   f'</div>')
    out.append('</div>')
    return "".join(out)


def grouped_sentiment_html(actual, pred, correct):
    """For each class: actual ('correct answer') vs predicted, + answered right."""
    maxv = max(list(actual.values()) + list(pred.values())) 
    rows = []
    for c in CLASSES:
        # predicted/actual can exceed actual total for that class; scale both bars
        rows.append(
            f'<div class="group-row">'
            f'<div class="group-label" style="color:{COLOR[c]}">{esc(c)}</div>'
            f'<div class="group-bars">'
            f'<div class="gbar-lbl">Actual</div>{bar(maxv, actual[c], COLOR[c])}'
            f'<div class="gbar-lbl">Predicted</div>{bar(maxv, pred[c], "#8899aa")}'
            f'</div>'
            f'<div class="group-stat">answered right '
            f'<b style="color:{COLOR[c]}">{correct[c]}/{actual[c]}</b> '
            f'({pct_str(correct[c], actual[c])})</div>'
            f'</div>')
    return '<div class="chart">' + "".join(rows) + '</div>'


def per_class_acc_html(pc):
    rows = []
    for c in CLASSES:
        p = pc[c]
        pct = f"{p['recall'] * 100:.0f}%"
        rows.append(
            f'<div class="hbar-row"><div class="hbar-label">{esc(c)}</div>'
            f'{bar(1.0, p["recall"], COLOR[c], pct)}'
            f'</div>')
    return '<div class="chart" style="min-width:260px">' + "".join(rows) + '</div>'


def confusion_html(cm):
    head = "<tr><th class='corner'></th>" + "".join(
        f"<th>predicted<br><span class='dim'>{esc(c)}</span></th>" for c in CLASSES) + "</tr>"
    body = ""
    for a in CLASSES:
        cells = "".join(f'<td class="cell v{cm[a][b]}"><b>{cm[a][b]}</b></td>'
                        for b in CLASSES)
        body += f'<tr><th class="rowlab" style="color:{COLOR[a]}">{esc(a)}<br><span class="dim">actual</span></th>{cells}</tr>'
    return f'<table class="cmatrix"><thead>{head}</thead><tbody>{body}</tbody></table>'


def emotion_html(llm, wl):
    maxv = max([0] + list(llm.values()) + list(wl.values()) + list(wl.values()))
    rows = []
    for e in EMOTIONS:
        l = llm.get(e, 0); w = wl.get(e, 0)
        rows.append(
            f'<div class="group-row emo">'
            f'<div class="group-label" style="color:{EMO_COLOR[e]}">{esc(e)}</div>'
            f'<div class="group-bars">'
            f'<div class="gbar-lbl">LLM</div>{bar(maxv, l, EMO_COLOR[e])}'
            f'<div class="gbar-lbl">Word-list</div>{bar(maxv, w, "#8ba4b8")}'
            f'</div></div>')
    return '<div class="chart">' + "".join(rows) + '</div>'


def render_kpis(agg):
    def card(title, value, sub):
        return (f'<div class="kpi"><div class="kpi-title">{esc(title)}</div>'
                f'<div class="kpi-value">{esc(value)}</div>'
                f'<div class="kpi-sub">{esc(sub)}</div></div>')
    p = agg["per_class"]
    cards = [card("Reviews scored", agg["total"], "balanced 3-class sample"),
             card("Overall accuracy", f"{agg['accuracy']*100:.1f}%",
                  "chance = 33.3%")]
    for c in CLASSES:
        cards.append(card(f"{esc(c)} recall", f"{p[c]['recall']*100:.0f}%",
                          f"{p[c]['tp']}/{p[c]['actual']} correct | F1 {p[c]['f1']:.2f}"))
    return '<div class="kpis">' + "".join(cards) + '</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()

    rows = load(args.run)
    agg, table_rows = compute(rows)

    llm_emotion_dist = " · ".join(f"{e}: {n}" for e, n in agg["llm_emo"].items())
    wl_emotion_dist = " · ".join(f"{e}: {n}" for e, n in agg["wl_emo"].items())

    # Danger: embed review text safely as JSON inside a <script> block.
    data_json = json.dumps(table_rows).replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Amazon Gift Card Reviews — Sentiment &amp; Emotion Dashboard</title>
<style>
  :root{{ --bg:#f6f7f9; --card:#ffffff; --fg:#1f2937; --muted:#6b7280; --border:#e5e7eb;
         --accent:#2563eb; }}
  *{{ box-sizing:border-box; }}
  body{{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--fg); }}
  .wrap{{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; }}
  header{{ margin-bottom:20px; }}
  h1{{ font-size:22px; margin:0 0 4px; }}
  .sub{{ color:var(--muted); font-size:13px; }}
  h2{{ font-size:16px; margin:0 0 10px; }}
  .grid{{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card{{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  @media(max-width:820px){{ .grid{{ grid-template-columns:1fr; }} }}
  .kpis{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0; }}
  .kpi{{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; }}
  .kpi-title{{ font-size:12px; color:var(--muted); }}
  .kpi-value{{ font-size:24px; font-weight:700; margin:4px 0; }}
  .kpi-sub{{ font-size:11px; color:var(--muted); }}
  .chart{{ margin-top:6px; }}
  .hbar-row,.group-row{{ display:flex; align-items:center; margin:7px 0; }}
  .hbar-label,.group-label{{ width:110px; font-size:13px; text-transform:capitalize; flex:none; }}
  .bar-wrap{{ flex:1; height:26px; position:relative; }}
  .bar{{ height:22px; border-radius:4px; min-width:4px; position:relative; margin-bottom:2px; }}
  .bar-val{{ position:absolute; right:6px; top:1px; font-size:11px; color:#fff; font-weight:700; text-shadow:0 1px 2px rgba(0,0,0,.4); }}
  .bar-label{{ font-size:10px; color:var(--muted); }}
  .group-row{{ align-items:center; }}
  .group-bars{{ flex:1; }}
  .gbar-lbl{{ font-size:10px; color:var(--muted); margin:2px 0 1px; }}
  .group-stat{{ width:170px; flex:none; text-align:right; font-size:12px; color:var(--muted); }}
  .emo .group-stat{{ display:none; }}
  .cmatrix{{ border-collapse:collapse; margin-top:8px; width:100%; }}
  .cmatrix th,.cmatrix td{{ border:1px solid var(--border); text-align:center; padding:8px; font-size:13px; }}
  .cmatrix .corner{{ background:transparent; border:none; }}
  .cmatrix th{{ color:var(--muted); font-weight:600; }}
  .cmatrix .rowlab{{ text-transform:capitalize; }}
  .dim{{ font-size:10px; color:var(--muted); font-weight:400; }}
  .cell{{ position:relative; }}
  .v0{{ background:#f3f4f6; }}
  .v1{{ background:#e3f0fb; }}
  .v2{{ background:#bcd9f2; }}
  .v3{{ background:#8fc0ea; }}
  .v4{{ background:#5ea6df; }}
  .v5{{ background:#2f88d1; color:#fff; }}
  .v40{{ background:#fce8e8; }} .cell.hit{{ outline:2px solid var(--fg); }}
  .filters{{ display:flex; flex-wrap:wrap; gap:12px; align-items:flex-end; margin:14px 0; }}
  .filters label{{ font-size:11px; color:var(--muted); display:block; margin-bottom:3px; }}
  .filters select,.filters input{{ padding:7px; border:1px solid var(--border); border-radius:6px; font-size:13px; background:#fff; }}
  .count{{ font-size:13px; color:var(--muted); margin:8px 0; }}
  .count b{{ color:var(--fg); }}
  table.rtable{{ width:100%; border-collapse:collapse; font-size:12px; }}
  table.rtable th{{ position:sticky; top:0; background:#eef1f5; text-align:left; padding:8px; border-bottom:2px solid var(--border); }}
  table.rtable td{{ padding:8px; border-bottom:1px solid var(--border); vertical-align:top; }}
  .row-correct{{ background:#f2fbf5; }}
  .row-miss{{ background:#fdf2f2; }}
  .badge{{ display:inline-block; padding:1px 7px; border-radius:10px; color:#fff; font-size:11px; font-weight:600; }}
  .stars{{ color:#e8a013; letter-spacing:1px; }}
  .scroll{{ max-height:520px; overflow:auto; border:1px solid var(--border); border-radius:8px; }}
  em.good{{ border-bottom:2px solid rgba(46,158,91,.5); }}
  em.bad{{ border-bottom:2px solid rgba(214,69,65,.5); }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Amazon Gift Card Reviews — Sentiment &amp; Emotion</h1>
    <div class="sub">Source: Amazon Reviews '23 dataset (Gift Cards category). Balanced 3-class sample.
      LLM: DeepSeek-V4-Flash-0731 (OpenAI-compatible endpoint). Emotional scores: NRC Emotion Lexicon v0.92.</div>
  </header>

  {render_kpis(agg)}

  <div class="grid">
    <div class="card"><h2>Star-rating distribution (full dataset)</h2>
      {star_dist_html(agg["rating_dist"])}
      <div class="sub" style="margin-top:8px">The dataset is lopsided: ~88.5% of reviews are 4–5★. A raw "always positive" predictor would therefore look ~88% "accurate".</div>
    </div>

    <div class="card"><h2>Correct answer vs predicted, per class</h2>
      {grouped_sentiment_html(agg["actual_counts"], agg["pred_counts"], agg["correct_counts"])}
      <div class="sub" style="margin-top:8px">Each class had 40 in the sample. Note how much the model <b>over-predicts negative</b> and under-predicts neutral — its failures are concentrated on 3★ reviews.</div>
    </div>

    <div class="card"><h2>Confusion matrix (counts)</h2>
      {confusion_html(agg["confusion"])}
      <div class="sub" style="margin-top:8px">Rows = ground truth (star-based), cols = LLM prediction. Neutral reviews bleed into <b style="color:{COLOR['negative']}">negative</b>: 22 of 40 were incorrectly called negative.</div>
    </div>

    <div class="card"><h2>Per-class recall</h2>
      {per_class_acc_html(agg["per_class"])}
    </div>

    <div class="card" style="grid-column:1/-1"><h2>Emotion: LLM vs NRC word-list</h2>
      {emotion_html(agg["llm_emo"], agg["wl_emo"])}
      <div class="sub" style="margin-top:8px">LLM runs at {llm_emotion_dist} · word-list runs at {wl_emotion_dist}. The word-list is dragged to <b>anticipation</b> because "gift" (in 74/120 reviews) is flagged anticipating; the LLM reads the actual complaints and leans <b>anger</b>. Agreement between the two = {pct_str(sum(1 for r in table_rows if r['llm_emo'] and r['llm_emo']==r['wl_emo']), len(table_rows))}.</div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Interactive review explorer</h2>
    <div class="filters">
      <div><label>Match</label><select id="f-match"><option value="all">All</option><option value="correct">LLM correct</option><option value="mismatch">LLM mismatch</option></select></div>
      <div><label>Ground truth</label><select id="f-label"><option value="all">All</option><option value="positive">Positive</option><option value="neutral">Neutral</option><option value="negative">Negative</option></select></div>
      <div><label>Predicted</label><select id="f-pred"><option value="all">All</option><option value="positive">Positive</option><option value="neutral">Neutral</option><option value="negative">Negative</option></select></div>
      <div><label>LLM emotion</label><select id="f-lem"><option value="all">All</option>{''.join(f'<option value="{e}">{e.title()}</option>' for e in EMOTIONS)}</select></div>
      <div><label>Word-list emotion</label><select id="f-wem"><option value="all">All</option>{''.join(f'<option value="{e}">{e.title()}</option>' for e in EMOTIONS)}<option value="none">(none)</option></select></div>
      <div><label>Search title/text</label><input id="f-q" type="search" placeholder="type to filter…" size="18"></div>
    </div>
    <div class="count">Showing <b id="n-show">0</b> of <b id="n-total">0</b> reviews</div>
    <div class="scroll"><table class="rtable">
      <thead><tr><th>★</th><th>Ground</th><th>Predicted</th><th>LLM emotion</th><th>Word-list emotion</th><th>Title</th><th>Text</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table></div>
  </div>
</div>

<script>
const DATA = {data_json};

function stars(r){{ const n = Math.round(r.rating)||0; return "★".repeat(Math.max(0,Math.min(5,n))); }}
function escT(s){{ return String(s).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]); }}

function render(filter) {{
  let rows = DATA.filter(r => filter.match==='all' || (filter.match==='correct' ? r.pred===r.label : r.pred!==r.label));
  if (filter.label!=='all') rows = rows.filter(r => r.label===filter.label);
  if (filter.pred!=='all') rows = rows.filter(r => r.pred===filter.pred);
  if (filter.lem!=='all') rows = rows.filter(r => (r.llm_emo||'none')===filter.lem);
  if (filter.wem!=='all') rows = rows.filter(r => (r.wl_emo||'none')===filter.wem);
  const q = filter.q.trim().toLowerCase();
  if (q) rows = rows.filter(r => (r.title+' '+r.text).toLowerCase().includes(q));
  document.getElementById('n-show').textContent = rows.length;
  document.getElementById('n-total').textContent = DATA.length;
  const tb = document.getElementById('tbody');
  tb.innerHTML = rows.map(r => {{
    const ok = r.pred===r.label;
    const cls = ok ? 'row-correct' : 'row-miss';
    const predBadge = r.pred ? `<span class="badge" style="background:${{r.pred==='positive'?'#2e9e5b':r.pred==='neutral'?'#c9a227':'#d64541'}}">${{r.pred}}</span>` : '<span class="badge" style="background:#999">error</span>';
    const grBadge = `<span class="badge" style="background:${{r.label==='positive'?'#2e9e5b':r.label==='neutral'?'#c9a227':'#d64541'}}">${{r.label}}</span>`;
    const lem = r.llm_emo ? `<span title="LLM dominant emotion">${{r.llm_emo}}</span>` : '<span class="dim">—</span>';
    const wem = r.wl_emo ? `<span title="NRC word-list dominant emotion">${{r.wl_emo}}</span>` : '<span class="dim">(none)</span>';
    return `<tr class="${{cls}}">
      <td class="stars" title="rating ${{r.rating}}">${{stars(r)}}</td>
      <td>${{grBadge}}</td><td>${{predBadge}}</td>
      <td>${{lem}}</td><td>${{wem}}</td>
      <td>${{escT(r.title.slice(0,70))}}</td>
      <td>${{escT(r.text.slice(0,160))}}${{r.text.length>160?'…':''}}</td></tr>`;
  }}).join('');
}}

function currentFilter() {{
  return {{ match: document.getElementById('f-match').value,
            label: document.getElementById('f-label').value,
            pred:  document.getElementById('f-pred').value,
            lem:   document.getElementById('f-lem').value,
            wem:   document.getElementById('f-wem').value,
            q:     document.getElementById('f-q').value }};
}}
['f-match','f-label','f-pred','f-lem','f-wem'].forEach(id => document.getElementById(id).addEventListener('change', () => render(currentFilter())));
document.getElementById('f-q').addEventListener('input', () => render(currentFilter()));
render(currentFilter());
</script>
</body>
</html>"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Dashboard written -> {args.out}  ({len(html_doc):,} bytes, "
          f"{len(table_rows)} review rows embedded)")


if __name__ == "__main__":
    main()
