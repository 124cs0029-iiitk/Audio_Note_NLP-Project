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
SUM_NAME = "philschmid/bart-large-cnn-samsum"          # the summarizer (meetings and lectures)

CHUNK_TOKENS = 300          # first level: small pieces, so details survive
GROUP_TOKENS = 700          # higher levels: group the partial summaries
MAX_SUMMARY_WORDS = 250     # stop summarizing again when the text is this short
RULES_VERSION = "tasks-v4"  # shown under the title of every report: tells which code made it

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
    r"\b(?P<who>I|we)\s*(?:(?:still|also|just|really|definitely|probably)\s+)?"
    r"(?P<modal>will|'ll|am going to|'m going to|can|need to|have to|must|should)\s+(?P<task>.+)$", re.I)
LETS_RE = re.compile(r"\b(?:let's|let us)\s+(.+)$", re.I)
# Whisper often drops the 's: "let schedule another meeting"
LET_PLAN_RE = re.compile(r"\blet\s+((?:schedule|arrange|plan|set up|book|review|discuss|meet)\b.+)$", re.I)
ASSIGN_RE = re.compile(
    r"\b([A-Za-z]+),?\s+(?:please|(?:can|could|would|will) you(?: also)?)\s+(.+)$", re.I)
SKIP_TASK_RE = re.compile(r"^(?:start|begin|get started|move on|continue|go through)\b", re.I)
# "I can see / hear / understand ..." is not a promise
NOT_A_PROMISE = {"see", "hear", "understand", "imagine", "believe", "think", "tell", "say", "assure"}
EMPTY_ENDING = {"it", "this", "that", "them", "there", "so", "too"}   # "review it" says nothing


def _too_short(task):
    words = task.split()
    return len(words) <= 2 and words[-1].lower().strip(".,!?") in EMPTY_ENDING


def _sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def _dates(sentence):
    return [d.group(0) for d in DATE_RE.finditer(sentence)]


def _join_cut_segments(segs):
    """A sentence that was cut at the end of one segment ("... the remaining tasks are...") goes on
    in the next segment of the same speaker: join them, so that the rules see the whole sentence."""
    out = []
    for seg in segs:
        prev = out[-1] if out else None
        if prev and prev["speaker"] == seg["speaker"] and prev["text"].rstrip().endswith(("...", "\u2026")):
            head = prev["text"].rstrip().rstrip(".\u2026").rstrip()
            tail = seg["text"].strip()
            first = tail.split()[0] if tail else ""
            if first and first[0].isupper() and not first.isupper() and not first.startswith("I'"):
                tail = tail[0].lower() + tail[1:]
            prev["text"] = head + " " + tail
            prev["end"] = seg.get("end", prev.get("end"))
        else:
            out.append(dict(seg))
    return out


# where a new statement starts inside a sentence: ", but please ...", " and I ...", "; ..."
NEW_STATEMENT_RE = re.compile(
    r",\s*(?:but|and)\s+(?=(?:please|I|we|you|they|he|she|it|the|this|that|there)\b)"
    r"|\s+(?:and|but)\s+(?=(?:I|we|you|please)\b)"
    r"|;\s*", re.I)
# where the text of a task ends: at a new statement, or at a reason ("so that we ...", "because ...")
TASK_END_RE = re.compile(
    NEW_STATEMENT_RE.pattern + r"|,?\s+so\s+(?=(?:that|I|we|you)\b)|,\s*(?:because|which)\b", re.I)


def _trim_task(task):
    """'arrange a call tomorrow, but please make sure ...' -> 'arrange a call tomorrow'"""
    first = TASK_END_RE.split(task.strip())[0].strip(" ,;")
    first = re.sub(r"\b(?:try|trying|do|doing)\s+(?:my|our)\s+best\s+to\s+", "", first, flags=re.I)
    return first if len(first.split()) >= 3 else task.strip()


# "Please send the file", "Could you share the logs?"  (a request, without anybody's name)
REQUEST_RE = re.compile(r"\b(?:(?:could|can|would|will) you(?: also)?(?: please)?|please)\s+(.+)$", re.I)
# requests that only ask for talk or give permission are not tasks
NOT_A_TASK_RE = re.compile(
    r"^(?:explain|tell|say|describe|clarify|repeat|mention|let me|go ahead|feel free|proceed|continue|"
    r"start|begin|move on|get started|go through|take your time|hold on|wait)\b", re.I)


def _name_owners(names):
    """first or last name -> the one speaker who has it (a name that two speakers share is left out)"""
    owners = defaultdict(set)
    for n in names:
        for w in re.findall(r"[A-Za-z]{3,}", display_name(n)):
            owners[w.lower()].add(n)
    return {w: next(iter(s)) for w, s in owners.items() if len(s) == 1}


def extract_actions(segs):
    """Find who has to do what, and by when. Every statement of a sentence is looked at on its own."""
    segs = _join_cut_segments(segs)
    names = {s["speaker"] for s in segs}
    owner_of_name = _name_owners(names)
    items, last_assign = [], None

    for i, seg in enumerate(segs):
        others = [n for n in names if n != seg["speaker"]]
        for sent in _sentences(seg["text"]):
            for clause in NEW_STATEMENT_RE.split(sent):
                clause = clause.strip()
                if not clause:
                    continue
                clean = clause.rstrip(".!?").strip()
                due = " ".join(_dates(clause))

                def new_item(owner, task):
                    return {"owner": owner, "task": _trim_task(task), "due": due,
                            "said_by": seg["speaker"], "at": seg["start"], "source": sent}

                # "Priya, please send ..." / "Anil, can you ...?"  (a first or last name of another speaker)
                m = ASSIGN_RE.search(clean)
                who = owner_of_name.get(m.group(1).lower()) if m else None
                if m and who and who != seg["speaker"] and not NOT_A_TASK_RE.match(m.group(2)):
                    item = new_item(who, m.group(2))
                    items.append(item)
                    last_assign = (i, item)
                    continue

                # "Please make sure ..." / "Could you share ...?"  (a request to the other person)
                m = REQUEST_RE.search(clean)
                if m and len(m.group(1).split()) >= 3 and not NOT_A_TASK_RE.match(m.group(1)):
                    owner = others[0] if len(others) == 1 else "Unassigned"
                    item = new_item(owner, m.group(1))
                    items.append(item)
                    last_assign = (i, item)
                    continue

                # "I will ..." / "I can ..." / "We need to ..."
                m = SELF_RE.search(clean)
                if m:
                    task = m.group("task")
                    first_word = task.split()[0].lower() if task.split() else ""
                    if m.group("modal").lower() == "can" and first_word in NOT_A_PROMISE:
                        continue                   # "I can see that ..." says nothing about work to do
                    if _too_short(task):
                        continue
                    # "Sure, I will send it": a short answer that accepts the request just made
                    accepted = (last_assign and i - last_assign[0] <= 1
                                and last_assign[1]["owner"] == seg["speaker"]
                                and ACCEPT_RE.match(clean) and len(task.split()) <= 6)
                    if accepted:
                        if not last_assign[1]["due"] and due:
                            last_assign[1]["due"] = due
                        continue
                    owner = seg["speaker"] if m.group("who").lower() == "i" else "Team"
                    items.append(new_item(owner, task))
                    continue

                # "Let's ..." / "let schedule ..."
                m = LETS_RE.search(clean) or LET_PLAN_RE.search(clean)
                if m and not SKIP_TASK_RE.match(m.group(1)) and not _too_short(m.group(1)):
                    items.append(new_item("Team", m.group(1)))
    return _remove_duplicates(items)


def _remove_duplicates(items):
    """The same task said twice (same owner, nearly the same words) is listed once."""
    out = []
    for it in items:
        words = set(re.findall(r"[a-z]+", it["task"].lower()))
        same = None
        for o in out:
            ow = set(re.findall(r"[a-z]+", o["task"].lower()))
            if o["owner"] == it["owner"] and words and ow and len(words & ow) / len(words | ow) >= 0.8:
                same = o
                break
        if same is None:
            out.append(it)
        elif not same["due"] and it["due"]:
            same["due"] = it["due"]
    return out


# a date counts as a "key date" when something is planned or due, or when it is an exact date
CUE_RE = re.compile(
    r"\b(?:before|by|until|till|due(?! to)|deadline|submit|submission|deliver|meeting|schedule|"
    r"arrange|call|release|launch|rehearsal|demo|presentation|exam|appointment)\b", re.I)
WEAK_DATE_RE = re.compile(r"^(?:today|tonight|this\s+(?:week|month|year|weekend))$", re.I)   # need a cue
PAST_DATE_RE = re.compile(r"^(?:yesterday|last\s+.+)$", re.I)                               # never


def _useful_dates(sentence):
    has_cue = bool(CUE_RE.search(sentence))
    out = []
    for d in _dates(sentence):
        if PAST_DATE_RE.match(d):
            continue
        if WEAK_DATE_RE.match(d) and not has_cue:
            continue
        out.append(d)
    return out


def extract_key_dates(segs):
    """Every sentence that mentions a date or time, and who said it."""
    out = []
    for seg in _join_cut_segments(segs):
        for sent in _sentences(seg["text"]):
            dates = _useful_dates(sent)
            if dates:
                out.append({"dates": dates, "text": sent,
                            "said_by": seg["speaker"], "at": seg["start"]})
    return out


# ---------- summary: small pieces first, then summaries of summaries ----------
def trim_to_sentence(text):
    """If the model ran out of room, its last sentence may be cut in the middle: drop it."""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[:cut + 1] if cut > 0 else text


def summarize(text, max_len=80, min_len=15):
    m = _load()
    tok, model = m["sum_tok"], m["sum"]
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=max_len, min_length=min_len,
                             num_beams=4, no_repeat_ngram_size=3, do_sample=False)
    out = tok.decode(ids[0], skip_special_tokens=True)
    if len(ids[0]) >= max_len:                  # it used all the room: the end may be cut off
        out = trim_to_sentence(out)
    return out


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


ALIASES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Robin", "Drew"]


def display_name(name):
    """'124CS0029 PARCHURI POOJITH KUMAR' -> 'Parchuri Poojith Kumar' (no ID code, no shouting)."""
    parts = [w for w in name.split() if not any(ch.isdigit() for ch in w)]
    text = " ".join(parts) or name
    return text.title() if (text.isupper() or text.islower()) else text


def summarize_meeting(segs, max_words=MAX_SUMMARY_WORDS, plain=False):
    """Summarize small pieces, then summarize the summaries until the text is short.
    plain=True (lecture mode): the text has no speaker names.
    The summarizer gets short first names instead of long account names, which it garbles;
    the real names are put back afterwards."""
    name_of = {}
    if plain:
        lines = [s["text"] for s in segs]
    else:
        used = " ".join(s["text"] for s in segs).lower()
        free = [a for a in ALIASES if a.lower() not in used]
        alias_of = {}
        for sp in dict.fromkeys(s["speaker"] for s in segs):
            if free:
                a = free.pop(0)
                alias_of[sp] = a
                name_of[a] = display_name(sp)
        lines = [f'{alias_of.get(s["speaker"], s["speaker"])}: {s["text"]}' for s in segs]
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
    text = " ".join(parts)
    for alias, real in name_of.items():             # put the real names back
        text = re.sub(rf"\b{alias}\b", real, text)
    return text


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


def lecture_summarize(text):
    """Shorten a text for lecture notes. It uses the summarizer that is already loaded
    for meetings, so no extra model is loaded (that used too much memory)."""
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
            "rules": RULES_VERSION,
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
        "rules": RULES_VERSION,
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