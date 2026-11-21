import importlib.util
import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "eval_summaries.json"
REFS = json.load(open(ROOT / "data" / "qmsum_refs.json", encoding="utf-8"))

# load nlp/report.py (the same code the server uses)
spec = importlib.util.spec_from_file_location("meeting_report", ROOT / "nlp" / "report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

meetings = sys.argv[1:] or ["qmsum_4", "qmsum_1"]
cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}

LEAD = "lead-100 (first 100 words)"
SINGLE = "single-pass (first 1024 tokens)"
CURRENT = "hierarchical (current)"
LONGER = "hierarchical (longer final)"
SYSTEMS = [LEAD, SINGLE, CURRENT, LONGER]


def save():
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def get(system, mid, make):
    """Return a cached result, or compute it (and save it right away)."""
    store = cache.setdefault(system, {})
    if mid not in store:
        print(f"  computing {system} for {mid}...", flush=True)
        store[mid] = make()
        save()
    return store[mid]


def climb(parts, max_len, min_len):
    """The higher levels of the hierarchy, starting from the first-level summaries."""
    while len(" ".join(parts).split()) > rep.MAX_SUMMARY_WORDS and len(parts) > 1:
        groups = rep.chunk_lines(parts, rep.GROUP_TOKENS)
        if len(groups) >= len(parts):
            break
        parts = [rep.summarize(g, max_len=max_len, min_len=min_len) for g in groups]
    return " ".join(parts)


for mid in meetings:
    print(f"{mid}:", flush=True)
    segs = rep.load_segments(mid)
    lines = [f'{s["speaker"]}: {s["text"]}' for s in segs]

    get(LEAD, mid, lambda: " ".join(" ".join(s["text"] for s in segs).split()[:100]))
    get(SINGLE, mid, lambda: rep.summarize("\n".join(lines), max_len=150, min_len=60))

    parts = get("_level0", mid, lambda: [
        rep.summarize(c, max_len=70, min_len=15)
        for c in rep.chunk_lines(lines, rep.CHUNK_TOKENS)])
    get(CURRENT, mid, lambda: climb(parts, 90, 30))
    get(LONGER, mid, lambda: climb(parts, 160, 60))

# ---------- scores ----------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
n = len(meetings)
ref_words = sum(len(REFS[m]["summary"].split()) for m in meetings) / n

print(f"\nROUGE F1 (%), average over {n} meeting(s): {', '.join(meetings)}")
print(f"human reference length: {ref_words:.0f} words on average\n")
print(f"{'system':34} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'words':>6}")
for name in SYSTEMS:
    r1 = r2 = rl = words = 0.0
    for mid in meetings:
        hyp = cache[name][mid]
        s = scorer.score(REFS[mid]["summary"], hyp)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
        words += len(hyp.split())
    print(f"{name:34} {100 * r1 / n:6.1f} {100 * r2 / n:6.1f} {100 * rl / n:6.1f} {words / n:6.0f}")