import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))     # so that lecture.py (same folder) can be imported

DB_PATH = Path(__file__).parent.parent / "data" / "meetings.db"
NER_NAME = "dslim/bert-base-NER"
SUM_NAME = "philschmid/bart-large-cnn-samsum"          # meetings (chat-style text)
LECTURE_SUM_NAME = "facebook/bart-large-cnn"            # lectures; "sshleifer/distilbart-cnn-12-6" is smaller

CHUNK_TOKENS = 300          # first level: small pieces, so details survive
GROUP_TOKENS = 700          # higher levels: group the partial summaries
MAX_SUMMARY_WORDS = 250     # stop summarizing again when the text is this short

_models = {}


def _load():
    """Load the models once and keep them in memory."""
    if _models:
        return _models
    print("Loading NER and summarization models (first run downloads them)...")
    _models["ner_tok"] = AutoTokenizer.from_pretrained(NER_NAME)
    _models["ner"] = AutoModelForTokenClassification.from_pretrained(NER_NAME).eval()
    _models["sum_tok"] = AutoTokenizer.from_pretrained(SUM_NAME)
    _models["sum"] = AutoModelForSeq2SeqLM.from_pretrained(SUM_NAME).eval()
    return _models


def load_segments(meeting_id):
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT speaker, start, end, text FROM segments "
        "WHERE meeting_id=? ORDER BY start", (meeting_id,)).fetchall()
    db.close()
    return [{"speaker": s, "start": st, "end": en, "text": t} for s, st, en, t in rows]


# ---------- entities: names, places, organizations (BERT) + dates, times (rules) ----------
TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+$")

DAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
         r"October|November|December)")
DATE_RE = re.compile(
    rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b"                  # October 28th
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}\b"       # 28th of October
    rf"|\b(?:next|this|last|every)\s+(?:{DAY}|week|month|year|weekend)\b"
    rf"|\b\d{{1,2}}(?::\d{{2}})?\s*(?:a\.m\.|p\.m\.|am|pm)(?!\w)"  # 10 AM, 3:30 pm
    rf"|\b(?:today|tomorrow|yesterday|tonight)\b"
    rf"|\b{DAY}\b",
    re.I)


def find_entities(text):
    """Return a list of (entity text, type)."""
    m = _load()
    tok, model = m["ner_tok"], m["ner"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256,
              return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    word_ids = enc.word_ids(0)
    inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    with torch.no_grad():
        pred = model(**inputs).logits[0].argmax(-1).tolist()

    # one label per whole word: the label of its first piece
    words = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        s, e = offsets[idx]
        if wid not in words:
            words[wid] = [s, e, model.config.id2label[pred[idx]]]
        else:
            words[wid][1] = e                      # the word continues over more pieces

    spans, cur = [], None
    for wid in sorted(words):
        s, e, lab = words[wid]
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        kind = lab[2:]
        if lab.startswith("B-") or cur is None or cur["type"] != kind:
            if cur:
                spans.append(cur)
            cur = {"type": kind, "start": s, "end": e}
        else:
            cur["end"] = e
    if cur:
        spans.append(cur)

    # keep titles such as Mr. / Dr. together with the name
    for x in spans:
        if x["type"] == "PER":
            t = TITLE_RE.search(text[:x["start"]])
            if t:
                x["start"] = t.start()

    found = [(text[x["start"]:x["end"]].strip(), x["type"]) for x in spans]

    # dates and times from simple rules
    dates = [(d.group(0), "DATE") for d in DATE_RE.finditer(text)]
    date_texts = {d.lower() for d, _ in dates}
    found = [(t, k) for t, k in found if t.lower() not in date_texts]
    return found + dates


# ---------- action items and key dates (rules) ----------
SENT_SPLIT = re.compile(
    r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<=[.?!])\s+")
ACCEPT_RE = re.compile(r"^(?:yes|yeah|sure|okay|ok|alright|of course|absolutely)\b", re.I)
SELF_RE = re.compile(
    r"\b(I|we)\s*(?:(?:still|also|just|really|definitely|probably)\s+)?"
    r"(?:will|'ll|am going to|'m going to|can|need to|have to|must|should)\s+(.+)$", re.I)
LETS_RE = re.compile(r"\b(?:let's|let us)\s+(.+)$", re.I)
ASSIGN_RE = re.compile(
    r"\b([A-Za-z]+),?\s+(?:please|(?:can|could|would|will) you(?: also)?)\s+(.+)$", re.I)
SKIP_TASK_RE = re.compile(r"^(?:start|begin|get started|move on|continue|go through)\b", re.I)


def _sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def _dates(sentence):
    return [d.group(0) for d in DATE_RE.finditer(sentence)]


def extract_actions(segs):
    """Find who has to do what, and by when."""
    names = {s["speaker"] for s in segs}
    items, last_assign = [], None

    for i, seg in enumerate(segs):
        for sent in _sentences(seg["text"]):
            clean = sent.rstrip(".!?").strip()
            due = " ".join(_dates(sent))

            def new_item(owner, task):
                return {"owner": owner, "task": task.strip(), "due": due,
                        "said_by": seg["speaker"], "at": seg["start"], "source": sent}

            # "Anil, please send ..." / "Anil, can you ...?"
            m = ASSIGN_RE.search(clean)
            if m and m.group(1) in names and m.group(1) != seg["speaker"]:
                item = new_item(m.group(1), m.group(2))
                items.append(item)
                last_assign = (i, item)
                continue

            # "I will ..." / "I can ..." / "We need to ..."
            m = SELF_RE.search(clean)
            if m:
                accepted = (last_assign and i - last_assign[0] <= 1
                            and last_assign[1]["owner"] == seg["speaker"]
                            and ACCEPT_RE.match(clean))
                if accepted:                       # "Sure, I will do it": task already listed
                    if not last_assign[1]["due"] and due:
                        last_assign[1]["due"] = due
                    continue
                owner = seg["speaker"] if m.group(1).lower() == "i" else "Team"
                items.append(new_item(owner, m.group(2)))
                continue

            # "Let's ..."
            m = LETS_RE.search(clean)
            if m and not SKIP_TASK_RE.match(m.group(1)):
                items.append(new_item("Team", m.group(1)))
    return items


def extract_key_dates(segs):
    """Every sentence that mentions a date or time, and who said it."""
    out = []
    for seg in segs:
        for sent in _sentences(seg["text"]):
            dates = [d for d in _dates(sent) if d.lower() != "yesterday"]
            if dates:
                out.append({"dates": dates, "text": sent,
                            "said_by": seg["speaker"], "at": seg["start"]})
    return out


# ---------- summary: small pieces first, then summaries of summaries ----------
def summarize(text, max_len=80, min_len=15):
    m = _load()
    tok, model = m["sum_tok"], m["sum"]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=max_len, min_length=min_len,
                             num_beams=4, no_repeat_ngram_size=3, do_sample=False)
    return tok.decode(ids[0], skip_special_tokens=True)


def chunk_lines(lines, budget):
    """Group lines into chunks of about `budget` tokens (a line is never cut)."""
    tok = _load()["sum_tok"]
    chunks, cur, used = [], [], 0
    for line in lines:
        n = len(tok(line, add_special_tokens=False)["input_ids"])
        if cur and used + n > budget:
            chunks.append("\n".join(cur))
            cur, used = [], 0
        cur.append(line)
        used += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def summarize_meeting(segs, max_words=MAX_SUMMARY_WORDS, plain=False):
    """Summarize small pieces, then summarize the summaries until the text is short.
    plain=True (lecture mode): the text has no speaker names."""
    if plain:
        lines = [s["text"] for s in segs]
    else:
        lines = [f'{s["speaker"]}: {s["text"]}' for s in segs]
    chunks = chunk_lines(lines, CHUNK_TOKENS)
    print(f"  level 0: {len(chunks)} pieces", flush=True)
    parts = [summarize(c, max_len=70, min_len=15) for c in chunks]

    level = 0
    while len(" ".join(parts).split()) > max_words and len(parts) > 1:
        level += 1
        groups = chunk_lines(parts, GROUP_TOKENS)
        if len(groups) >= len(parts):               # nothing to merge any more
            break
        print(f"  level {level}: {len(groups)} groups", flush=True)
        parts = [summarize(g, max_len=90, min_len=30) for g in groups]
    return " ".join(parts)


def lecture_notes(segs):
    """One speaker: a short note for each part of the talk, with the time it starts."""
    tok = _load()["sum_tok"]
    notes, cur, used, start = [], [], 0, None

    def flush():
        if not cur:
            return
        text = " ".join(cur)
        if len(text.split()) >= 30:             # a very short piece is kept as it is
            text = summarize(text, max_len=70, min_len=15)
        notes.append({"at": start, "text": text})

    for s in segs:
        n = len(tok(s["text"], add_special_tokens=False)["input_ids"])
        if cur and used + n > CHUNK_TOKENS:
            flush()
            cur, used, start = [], 0, None
        if start is None:
            start = s["start"]
        cur.append(s["text"])
        used += n
    flush()
    return notes


# ---------- gender-neutral wording with correct grammar ----------
ADVERBS = (r"(?:always|never|often|also|usually|sometimes|really|just|still|even|"
           r"only|already|now|not|perhaps)")

IRREGULAR = {
    "is": "are", "has": "have", "was": "were", "does": "do", "goes": "go",
    "isn't": "aren't", "hasn't": "haven't", "wasn't": "weren't", "doesn't": "don't",
}

OBJECT_VERBS = {
    "help", "helps", "helped", "helping", "let", "lets", "make", "makes", "made",
    "tell", "tells", "told", "ask", "asks", "asked", "give", "gives", "gave",
    "show", "shows", "showed", "teach", "teaches", "taught", "thank", "thanks",
    "thanked", "call", "calls", "called", "see", "sees", "saw", "meet", "meets",
    "met", "join", "joins", "joined", "invite", "invites", "invited", "allow",
    "allows", "allowed", "encourage", "encourages", "encouraged", "want", "wants",
    "wanted",
}

BASE_VERBS = {
    "pick", "get", "go", "do", "make", "feel", "be", "learn", "work", "find",
    "understand", "remember", "keep", "stay", "see", "take", "come", "play",
    "study", "write", "read", "cook", "clean", "build", "prepare", "manage",
    "improve", "relax", "focus", "practice", "speak", "talk", "use", "buy",
    "pay", "join", "finish", "start", "enjoy",
}


def plural(verb):
    """Turn a 3rd-person singular verb into the form that goes with 'they'."""
    low = verb.lower()
    if low in IRREGULAR:
        new = IRREGULAR[low]
    elif len(low) > 4 and low.endswith("ies"):
        new = low[:-3] + "y"                              # studies -> study
    elif low.endswith(("ches", "shes", "sses", "xes", "zes")):
        new = low[:-2]                                    # watches -> watch
    elif len(low) > 2 and low.endswith("s") and not low.endswith(("ss", "us", "is")):
        new = low[:-1]                                    # uses -> use
    else:
        return verb
    return new.capitalize() if verb[0].isupper() else new


def _subject(m):
    pron, adv, verb = m.group(1), m.group(2) or "", m.group(3)
    they = "They" if pron[0].isupper() else "they"
    return they + adv + " " + plural(verb)


def _her_is_object(text, start, end):
    """'told her.' / 'helps her pick' -> them;  'her team' -> their."""
    before = re.findall(r"[A-Za-z']+", text[max(0, start - 30):start])
    prev = before[-1].lower() if before else ""
    m = re.match(r"\s*([A-Za-z']+)", text[end:])
    nxt = m.group(1).lower() if m else ""
    if not nxt:
        return True
    return prev in OBJECT_VERBS and nxt in BASE_VERBS


def _other(m):
    word = m.group(0)
    low = word.lower()
    if low == "her":
        new = "them" if _her_is_object(m.string, m.start(), m.end()) else "their"
    else:
        new = {"his": "their", "hers": "theirs", "him": "them",
               "himself": "themselves", "herself": "themselves"}[low]
    return new.capitalize() if word[0].isupper() else new


def neutral(text):
    """Remove gender guesses made by the summarizer, keeping the grammar correct."""
    text = re.sub(r"\b(he|she)'s\b",
                  lambda m: ("They" if m.group(1)[0].isupper() else "they") + "'re",
                  text, flags=re.I)
    text = re.sub(rf"\b(he|she)\b((?:\s+{ADVERBS})*)\s+([A-Za-z']+)", _subject,
                  text, flags=re.I)
    text = re.sub(r"\b(he|she)\b",
                  lambda m: "They" if m.group(1)[0].isupper() else "they",
                  text, flags=re.I)
    text = re.sub(r"\b(his|hers|her|him|himself|herself)\b", _other, text, flags=re.I)
    return text


def _lecture_model():
    """The summarizer for lectures. If it cannot be loaded, the meeting summarizer is used."""
    if "lec_sum" not in _models:
        try:
            print(f"Loading the lecture summarizer {LECTURE_SUM_NAME} (first run downloads it)...")
            _models["lec_tok"] = AutoTokenizer.from_pretrained(LECTURE_SUM_NAME)
            _models["lec_sum"] = AutoModelForSeq2SeqLM.from_pretrained(LECTURE_SUM_NAME).eval()
        except Exception as e:
            print("Could not load it, using the meeting summarizer instead:", e)
            m = _load()
            _models["lec_tok"], _models["lec_sum"] = m["sum_tok"], m["sum"]
    return _models["lec_tok"], _models["lec_sum"]


def lecture_summarize(text):
    """Rewrite the key sentences of one topic into a shorter text.
    Uses the summarizer that is already loaded (no extra memory)."""
    return summarize(text, max_len=90, min_len=25)

def build_report(segs, mode="meeting"):
    """mode "meeting": summary, action items, key dates, who said what.
    mode "lecture": one speaker (a talk or video): clean the text, split it into topics, pick
    the key sentences, rewrite them, and check every sentence against the transcript."""
    if mode == "lecture":
        import lecture as lecture_mod                  # nlp/lecture.py
        res = lecture_mod.build_lecture(segs, lecture_summarize)
        return {
            "mode": "lecture",
            "summary": res["summary"],
            "summary_items": res["summary_items"],
            "notes": res["notes"],
            "entities": res["keyterms"],
            "faithfulness": res["faithfulness"],
            "action_items": [],
            "key_dates": extract_key_dates(segs),
            "speakers": {},
            "segments": segs,
        }

    # 1. entities per segment
    for s in segs:
        s["entities"] = [{"text": t, "type": k} for t, k in find_entities(s["text"])]

    # 2. summary (small pieces, then summaries of summaries)
    summary = neutral(summarize_meeting(segs))

    # 3. who said what
    speakers = defaultdict(lambda: {"segments": 0, "talk_seconds": 0.0,
                                    "entities": set(), "lines": []})
    for s in segs:
        r = speakers[s["speaker"]]
        r["segments"] += 1
        r["talk_seconds"] += s["end"] - s["start"]
        r["lines"].append(s["text"])
        for e in s["entities"]:
            r["entities"].add(f'{e["text"]} ({e["type"]})')

    return {
        "mode": "meeting",
        "summary": summary,
        "notes": [],
        "action_items": extract_actions(segs),
        "key_dates": extract_key_dates(segs),
        "speakers": {
            name: {"segments": r["segments"],
                   "talk_seconds": round(r["talk_seconds"], 1),
                   "entities": sorted(r["entities"]),
                   "lines": r["lines"]}
            for name, r in speakers.items()
        },
        "segments": segs,
    }


def print_actions(actions, dates):
    print("ACTION ITEMS:")
    for a in actions:
        due = f"   (due: {a['due']})" if a["due"] else ""
        print(f"  - {a['owner']}: {a['task']}{due}   [said by {a['said_by']}]")
    print("\nKEY DATES:")
    for d in dates:
        print(f"  - {', '.join(d['dates'])}: {d['text']}   ({d['said_by']})")


if __name__ == "__main__":
    if sys.argv[1] == "--test-grammar":
        examples = [
            "Interviewer has a pit bull puppy at home. He is friendly and well behaved. "
            "Interviewer's puppy helps her pick up tennis balls.",
            "Student is an accountant. She uses a computer to manage business records. "
            "He also uses a smartphone.",
            "Student studies at university and he has a laptop. She doesn't work without it.",
            "He always wags his tail and she loves him.",
        ]
        for text in examples:
            print("before:", text)
            print("after :", neutral(text), "\n")
        sys.exit()

    if sys.argv[1] == "--actions":               # quick test, no models needed
        segs = load_segments(sys.argv[2])
        print_actions(extract_actions(segs), extract_key_dates(segs))
        sys.exit()

    if sys.argv[1] == "--summary":               # summary only, with timing
        import time
        t0 = time.time()
        segs = load_segments(sys.argv[2])
        n_words = sum(len(s["text"].split()) for s in segs)
        print(f"{len(segs)} lines, {n_words} words")
        text = summarize_meeting(segs)
        print(f"\nSUMMARY ({len(text.split())} words, {time.time() - t0:.0f} s):\n", neutral(text))
        refs_file = DB_PATH.parent / "qmsum_refs.json"
        if refs_file.exists():
            refs = json.load(open(refs_file, encoding="utf-8"))
            if sys.argv[2] in refs:
                ref = refs[sys.argv[2]]["summary"]
                print(f"\nHUMAN REFERENCE ({len(ref.split())} words):\n", ref)
        sys.exit()

    meeting_id = sys.argv[1]
    report = build_report(load_segments(meeting_id))

    print("\nSUMMARY:\n", report["summary"], "\n")
    print_actions(report["action_items"], report["key_dates"])
    print()
    for name, r in report["speakers"].items():
        print(f"=== {name} ===")
        print(f"Segments: {r['segments']}   Talk time: {r['talk_seconds']} s")
        print("Entities:", ", ".join(r["entities"]) or "none")
        for line in r["lines"]:
            print("  -", line)
        print()