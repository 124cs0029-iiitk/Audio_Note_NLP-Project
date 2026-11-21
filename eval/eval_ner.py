import importlib.util
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("meeting_report", ROOT / "nlp" / "report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

MEETING = "demo1"

# Hand-made answer key: one list per line of demo1 (same order as make_demo_meeting.py).
# Types: PER person, ORG organization, LOC place, MISC product/other, DATE date or time.
GOLD = [
    [("Campus Connect", "MISC")],                                                    # 0
    [("Ravi", "PER"), ("Firebase", "ORG"), ("yesterday", "DATE")],                   # 1
    [("Android", "MISC"), ("iPhone", "MISC")],                                       # 2
    [("Android", "MISC"), ("iPhone", "MISC")],                                       # 3
    [("iPhone", "MISC"), ("Thursday", "DATE"), ("Hyderabad", "LOC")],                # 4
    [("Anil", "PER"), ("Priya", "PER"), ("Friday", "DATE")],                         # 5
    [("Friday", "DATE")],                                                            # 6
    [("Razorpay", "ORG")],                                                           # 7
    [("Razorpay", "ORG"), ("today", "DATE")],                                        # 8
    [],                                                                              # 9
    [("Dr. Sharma", "PER")],                                                         # 10
    [("Dr. Sharma", "PER"), ("Chennai", "LOC"), ("October 28th", "DATE")],           # 11
    [("Monday", "DATE")],                                                            # 12
    [("Anil", "PER")],                                                               # 13
    [("next week", "DATE")],                                                         # 14
    [],                                                                              # 15
    [],                                                                              # 16
    [],                                                                              # 17
    [],                                                                              # 18
    [("Priya", "PER"), ("Wednesday", "DATE")],                                       # 19
    [("Wednesday", "DATE")],                                                         # 20
    [("Microsoft Azure", "ORG"), ("next month", "DATE")],                            # 21
    [],                                                                              # 22
    [("IT office", "ORG")],                                                          # 23
    [("Firebase", "ORG")],                                                           # 24
    [],                                                                              # 25
    [],                                                                              # 26
    [("next Monday", "DATE"), ("10 AM", "DATE")],                                    # 27
]

segs = rep.load_segments(MEETING)
if len(segs) != len(GOLD):
    sys.exit(f"{MEETING} has {len(segs)} lines but the answer key has {len(GOLD)}. "
             "Run python nlp/make_demo_meeting.py first.")


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def stripped(text):
    """Lowercase and no punctuation, like an uncased speech recognizer."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(preds, strict):
    tp = fp = fn = 0
    by_type = defaultdict(lambda: [0, 0, 0])
    for p, g in zip(preds, GOLD):
        pc = Counter((norm(t), k if strict else "") for t, k in p)
        gc = Counter((norm(t), k if strict else "") for t, k in g)
        common, extra, missed = pc & gc, pc - gc, gc - pc
        tp += sum(common.values())
        fp += sum(extra.values())
        fn += sum(missed.values())
        for key, n in common.items():
            by_type[key[1]][0] += n
        for key, n in extra.items():
            by_type[key[1]][1] += n
        for key, n in missed.items():
            by_type[key[1]][2] += n
    return (tp, fp, fn), by_type


print(f"Running NER on {len(segs)} lines of {MEETING}...", flush=True)
original = [rep.find_entities(s["text"]) for s in segs]
lowered = [rep.find_entities(stripped(s["text"])) for s in segs]

print(f"\n{'text':9} {'match':8} {'P':>5} {'R':>5} {'F1':>5}   tp/fp/fn")
for name, preds in [("original", original), ("no-punct", lowered)]:
    for mode, strict in [("strict", True), ("lenient", False)]:
        (tp, fp, fn), _ = score(preds, strict)
        p, r, f = prf(tp, fp, fn)
        print(f"{name:9} {mode:8} {p:5.2f} {r:5.2f} {f:5.2f}   {tp}/{fp}/{fn}")

print("\nPer type, strict, original text:")
print(f"{'type':6} {'P':>5} {'R':>5} {'F1':>5}   tp/fp/fn")
_, by_type = score(original, True)
for k in sorted(by_type):
    tp, fp, fn = by_type[k]
    p, r, f = prf(tp, fp, fn)
    print(f"{k:6} {p:5.2f} {r:5.2f} {f:5.2f}   {tp}/{fp}/{fn}")

print("\nMistakes on the original text (strict):")
for i, (p, g) in enumerate(zip(original, GOLD)):
    pc = Counter((norm(t), k) for t, k in p)
    gc = Counter((norm(t), k) for t, k in g)
    missed, extra = gc - pc, pc - gc
    if missed or extra:
        print(f"  line {i}: missed {list(missed)}  extra {list(extra)}")