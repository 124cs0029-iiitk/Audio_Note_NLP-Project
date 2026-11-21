"""Lecture mode: clean the transcript, split it into topics, pick the key sentences of each
topic, rewrite them with a summarizer, and check every summary sentence against the transcript.

Everything here is plain Python + NumPy. The summarizer model is passed in as a function,
so this file can be tested without any model."""
import math
import re
from collections import Counter

import numpy as np

STOP = set("""a an the and or but if so then than that this these those it its is are was were be been being am
to of in on at for from by with as into about over after before up down out off not no nor too very can could
will would shall should may might must do does did done have has had having i me my we us our you your he she
they them their there here what which who whom whose when where why how all any some each every more most other
such only own same just also now well ok okay alright yeah yes right like get got go going let lets really
actually basically thing things kind lot""".split())

FILLER_START = re.compile(
    r"^(?:so|now|alright|all right|okay|ok|and|but|well|yeah|then|first of all|and also)\b[,.]?\s+", re.I)
SENT_END = re.compile(r"(?<=[.?!])\s+")
DEFINITION = re.compile(r"\b(?:means|mean by|known as|refers to|is called|is defined|is a|is the|stands for)\b", re.I)

GENERIC = set("""provided important next between different many called given known example examples
used use using means mean take make made come comes said say call needs need thing things sometimes
service services course lecture series system systems""".split())
DANGLING = set("""a an the and or but of to in on at for with by from that which who is are was were so as
also it its this these those can will would should must may might if then than when while because""".split())
TOPIC_CUE = re.compile(
    r"\b(?:next|first|second|third|fourth|another|final|last|following)\b.{0,50}?"
    r"\b(?:is|are|topic|service|section|part|point|step|one)\b", re.I)
NOTE_TOPIC_START = re.compile(
    r"^(?:\d+[.)]\s*|\d+\s*$|next on our list\b|another question\b|here(?:['’]s| is) another question\b|"
    r"moving on to\b|turning to\b|let['’]s start with\b)", re.I)

SUPPORTED = 0.75      # a summary sentence with at least this overlap with the transcript is fine
WEAK = 0.50           # below this the sentence is replaced by a real transcript sentence


# ---------------- small text helpers ----------------
def stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def words(text):
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)]


def content_words(text):
    out = []
    for w in words(text):
        w = w.split("'")[0]
        if w and w not in STOP and (len(w) > 2 or w.isdigit()):
            out.append(stem(w))
    return out


def capfirst(s):
    return s[:1].upper() + s[1:] if s else s


# ---------------- 1. clean ----------------
def split_sentences(segs):
    """Sentences with the time they start. A sentence cut in the middle by the end of a
    segment is joined with the next segment only if that one continues in lower case."""
    out, carry, carry_at = [], "", None

    def flush_carry():
        nonlocal carry, carry_at
        if carry and len(carry.split()) >= 10:       # a long unfinished sentence is still useful
            out.append({"text": carry.rstrip(",;:") + ".", "at": carry_at})
        carry, carry_at = "", None

    for seg in segs:
        raw = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), seg["text"].strip())
        text = re.sub(r"(?:\s*\.){2,}\s*$", "", raw)       # a trailing "..." means: cut off
        cut_off = text != raw
        pieces = [p.strip() for p in SENT_END.split(text) if p.strip()]
        for i, p in enumerate(pieces):
            if carry:
                if p[:1].islower():                    # the cut sentence continues here
                    p, at = carry + " " + p, carry_at
                    carry, carry_at = "", None
                else:                                  # it was never finished
                    flush_carry()
                    at = seg["start"]
            else:
                at = seg["start"]
            last = i == len(pieces) - 1
            if last and (cut_off or not re.search(r"[.?!][\"')]?$", p)):
                carry, carry_at = p, at
            else:
                out.append({"text": p, "at": at})
    flush_carry()
    return out


def clean_sentences(sents):
    """Remove junk, leading filler words, and repeated sentences."""
    out, recent = [], []
    for s in sents:
        t = re.sub(r"(?:\s*\.){2,}", ".", s["text"]).strip()       # ". . ." -> "."
        for _ in range(2):
            t = FILLER_START.sub("", t).strip()
        if len(t.split()) < 5 or len(set(content_words(t))) < 3:     # "Speaker.", "And so on."
            continue
        last = re.sub(r"[^a-z]", "", t.split()[-1].lower())
        if not re.search(r"[.?!][\"')]?$", t) and last in DANGLING:  # "... which is also an"
            continue
        cw = set(content_words(t))
        dup = False
        for prev in recent[-6:]:
            if len(cw & prev) / max(1, len(cw | prev)) >= 0.8:
                dup = True
                break
        if dup:
            continue
        recent.append(cw)
        out.append({"text": capfirst(t), "at": s["at"]})
    return out


# ---------------- 2. vectors, topics, key sentences ----------------
def tfidf(sents):
    docs = [content_words(s["text"]) for s in sents]
    vocab = {}
    for d in docs:
        for w in d:
            vocab.setdefault(w, len(vocab))
    n = len(docs)
    df = Counter(w for d in docs for w in set(d))
    M = np.zeros((n, max(1, len(vocab))))
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            idf = math.log((n + 1) / (df[w] + 1)) + 1
            M[i, vocab[w]] = c * idf
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return M / norm, vocab


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def topic_split(vecs, sents, win=3, min_size=4, max_size=18):
    """Cut where the words of the next sentences differ most from the previous ones."""
    n = len(vecs)
    if n < 2 * min_size:
        return [(0, n)]
    sims = np.array([cosine(vecs[max(0, i - win):i].mean(0), vecs[i:i + win].mean(0))
                     for i in range(1, n)])
    thr = sims.mean() - 0.5 * sims.std()
    cuts = [0]
    for i in range(1, n):
        s = sims[i - 1]
        local_min = s <= sims[max(0, i - 2)] and s <= sims[min(len(sims) - 1, i)]
        cue = bool(TOPIC_CUE.search(sents[i]["text"][:90])) and i - cuts[-1] >= 3
        if (s < thr and local_min and i - cuts[-1] >= min_size) or cue or i - cuts[-1] >= max_size:
            cuts.append(i)
    if n - cuts[-1] < min_size and len(cuts) > 1:
        cuts.pop()                                  # join a tiny last piece to the one before
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


def key_sentences(sents, vecs, k):
    """TextRank: sentences that are similar to many others are central. Returns indexes in order."""
    n = len(sents)
    if n <= k:
        return list(range(n))
    S = vecs @ vecs.T
    np.fill_diagonal(S, 0)
    S[S < 0.05] = 0
    rows = S.sum(1, keepdims=True)
    rows[rows == 0] = 1
    P = S / rows
    r = np.ones(n) / n
    for _ in range(40):
        r = 0.15 / n + 0.85 * (P.T @ r)
    for i, s in enumerate(sents):
        wc = len(s["text"].split())
        if wc < 7:
            r[i] *= 0.6
        elif wc > 45:
            r[i] *= 0.8
        if s["text"].rstrip().endswith("?"):
            r[i] *= 0.6
        if DEFINITION.search(s["text"]):
            r[i] *= 1.2
    return sorted(np.argsort(-r)[:k].tolist())


# ---------------- 3. key terms ----------------
def _phrases(sents):
    """Count phrases of 1 to 3 content words. Returns (count, surface forms)."""
    surf, count = {}, Counter()
    for s in sents:
        run = []
        for tok in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.?!,;:]", s["text"]) + ["."]:
            low = tok.lower().split("'")[0]
            if re.fullmatch(r"[.?!,;:]", tok) or low in STOP or len(low) < 3:
                for n in (1, 2, 3):
                    for i in range(len(run) - n + 1):
                        gram = run[i:i + n]
                        key = " ".join(stem(g.lower()) for g in gram)
                        count[key] += 1
                        surf.setdefault(key, Counter())[" ".join(gram)] += 1
                run = []
            else:
                run.append(tok)
    return count, surf


def keyterms(sents, top=14):
    """Phrases that are repeated in the lecture (multi-word first), plus acronyms and a few words."""
    count, surf = _phrases(sents)
    n_sent = max(1, len(sents))
    df = Counter()
    for s in sents:
        for w in set(content_words(s["text"])):
            df[w] += 1
    multi, acro, single = [], [], []
    for key, c in count.items():
        kw = key.split()
        best = surf[key].most_common(1)[0][0]
        if any(w in GENERIC for w in (kw[0], kw[-1])):
            continue
        if len(kw) >= 2 and c >= 2:
            multi.append((c * (1 + 0.6 * (len(kw) - 1)), key, c, best))
        elif len(kw) == 1 and best.isupper() and 2 <= len(best) <= 5:
            acro.append((c, key, c, best))
        elif len(kw) == 1 and c >= 4 and len(key) >= 5 and df[key] / n_sent <= 0.2:
            single.append((c, key, c, best))
    multi.sort(reverse=True)
    chosen = []
    for item in multi:                                  # drop a phrase that sits inside a longer one
        kw = item[1].split()
        covered = False
        for other in chosen:
            ok = other[1].split()
            inside = any(ok[i:i + len(kw)] == kw for i in range(len(ok) - len(kw) + 1))
            if inside and len(ok) > len(kw) and other[2] >= 0.6 * item[2]:
                covered = True
                break
        if not covered:
            chosen.append(item)
    keep = []
    for item in chosen:
        kw = item[1].split()
        inside_other = False
        for other in chosen:
            ok = other[1].split()
            if other is not item and len(ok) > len(kw) and \
               any(ok[i:i + len(kw)] == kw for i in range(len(ok) - len(kw) + 1)) and \
               other[2] >= 0.5 * item[2]:
                inside_other = True
                break
        if not inside_other:
            keep.append(item)
    chosen = keep[:max(1, top - 5)]
    acro.sort(reverse=True)
    single.sort(reverse=True)
    terms = [capfirst(b) for _, _, _, b in chosen]
    terms += [b for _, _, _, b in acro[:3]]
    words_used = {w for _, k, _, _ in chosen for w in k.split()}
    terms += [capfirst(b) for _, k, _, b in single if k not in words_used][:2]
    out, seen = [], set()
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:top]


def cue_heading(sentence):
    """'The next service that is provided by the operating system is File System Manipulation.'
    -> 'File System Manipulation'"""
    m = None
    for m in re.finditer(r"\b(?:is known as|is called|is|are)\s+(?:the\s+|a\s+|an\s+)?", sentence):
        pass
    if not m:
        return ""
    tail = re.split(r"[.,?!;]| or | which | that | and ", sentence[m.end():])[0].strip()
    first = tail.split()[0].lower() if tail.split() else ""
    if 1 <= len(tail.split()) <= 5 and tail.lower() not in STOP and \
       first not in ("provided", "given", "used", "called", "known", "able", "one"):
        return capfirst(tail)
    return ""


def heading(part, terms):
    found = ""
    for s in part[:3]:                                  # the last cue wins (a repeated line comes first)
        if TOPIC_CUE.search(s["text"][:90]):
            found = cue_heading(s["text"]) or found
    if found:
        return found
    text = " ".join(s["text"].lower() for s in part)
    best, best_n = "", 1
    for t in terms:
        n = text.count(t.lower())
        if len(t.split()) >= 2 and n > best_n:
            best, best_n = t, n
    return best


def topic_keywords(sents, top=3):
    cw = Counter(w for s in sents for w in content_words(s["text"]))
    return [w for w, _ in cw.most_common(top)]


# ---------------- 4. faithfulness ----------------
def support(sentence, src, src_cw):
    """How much of the summary sentence's content is found in the transcript.
    Returns (score 0..1, time of the best matching place)."""
    need = set(content_words(sentence))
    if not need:
        return 1.0, None
    best, best_at = 0.0, None
    n = len(src)
    for i in range(n):
        have = set()
        for w in (1, 2, 3):
            if i + w > n:
                break
            have |= src_cw[i + w - 1]
            score = len(need & have) / len(need)
            if score > best + 1e-9:
                best, best_at = score, src[i]["at"]
    return best, best_at


def verify(text, src, src_cw):
    """Split a text into sentences and score each one."""
    items = []
    for sent in [x.strip() for x in SENT_END.split(text.strip()) if x.strip()]:
        score, at = support(sent, src, src_cw)
        items.append({"text": sent, "score": round(score, 2), "source_at": at})
    return items


# ---------------- the whole pipeline ----------------
def build_lecture(segs, summarize_fn, rewrite_min_words=25):
    """summarize_fn(text) -> shorter text (the summarizer model)."""
    sents = clean_sentences(split_sentences(segs))
    if not sents:
        return {"summary": "", "summary_items": [], "notes": [], "keyterms": [],
                "faithfulness": {"sentences": 0, "supported": 0, "flagged": 0, "average": 0.0,
                                 "rewrites_rejected": 0}}

    src_cw = [set(content_words(s["text"])) for s in sents]
    vecs, _ = tfidf(sents)
    notes, rejected = [], 0
    terms = keyterms(sents)
    for a, b in topic_split(vecs, sents):
        part, pv = sents[a:b], vecs[a:b]
        # Summarize a manageable selection of the central facts for this topic.
        # Preserve several points instead of the old two-or-three sentence limit.
        k = min(6, max(3, math.ceil(len(part) * 0.35)))
        picked = key_sentences(part, pv, k)
        extract = " ".join(part[i]["text"] for i in picked)
        method, items = "extractive", None

        if len(extract.split()) >= rewrite_min_words:
            rewritten = summarize_fn(extract)
            candidate_items = verify(rewritten, sents, src_cw)
            avg = sum(x["score"] for x in candidate_items) / max(1, len(candidate_items))
            if candidate_items and avg >= WEAK + 0.1 and min(x["score"] for x in candidate_items) >= WEAK:
                method = "rewritten"
                items = candidate_items
            else:
                rejected += 1
        if items is None:
            # Fallback: exact central sentences, still much shorter than the full transcript.
            items = verify(extract, sents, src_cw)
        notes.append({
            "at": part[0]["at"],
            "text": " ".join(x["text"] for x in items),
            "score": round(min(x["score"] for x in items), 2),
            "source_at": [x["source_at"] for x in items if x["source_at"] is not None],
            "heading": heading(part, terms),
            "keywords": topic_keywords(part),
            "method": method,
            "items": items,
        })

    # The summary contains every complete topic note, not just one lead sentence per topic.
    summary_items = []
    for nt in notes:
        for item in nt["items"]:
            summary_items.append(dict(item, at=nt["at"]))
    summary = " ".join(x["text"] for x in summary_items)

    every = [x for nt in notes for x in nt["items"]]
    supported = sum(1 for x in every if x["score"] >= SUPPORTED)

    # Keep every transcript sentence, grouped by topic so Notes read as coherent
    # points rather than one bullet per sentence. The Summary remains independent.
    actual_sentences = []
    for seg in segs:
        for text in re.split(r"(?<=[.!?])\s+", seg["text"].strip()):
            text = text.strip()
            if text:
                actual_sentences.append({"text": text, "at": seg["start"]})
    if actual_sentences:
        actual_vecs, _ = tfidf(actual_sentences)
        cuts = {0, len(actual_sentences)}
        for i, sentence in enumerate(actual_sentences):
            if i and NOTE_TOPIC_START.search(sentence["text"]):
                cuts.add(i)
        ordered_cuts = sorted(cuts)
        if len(ordered_cuts) == 2:
            note_groups = topic_split(actual_vecs, actual_sentences, max_size=15)
        else:
            note_groups = []
            for start, end in zip(ordered_cuts[:-1], ordered_cuts[1:]):
                part = actual_sentences[start:end]
                if len(part) > 18:
                    part_vecs, _ = tfidf(part)
                    note_groups.extend((start + a, start + b)
                                       for a, b in topic_split(part_vecs, part, max_size=15))
                else:
                    note_groups.append((start, end))
    else:
        note_groups = []
    notes = []
    for a, b in note_groups:
        part = actual_sentences[a:b]
        text = " ".join(s["text"].strip() for s in part if s["text"].strip())
        if text:
            notes.append({
                "at": None,
                "text": text,
                "score": 1.0,
                "source_at": [],
                "heading": "",
                "keywords": topic_keywords(part),
                "method": "transcript",
                "items": verify(text, sents, src_cw),
            })

    return {
        "summary": summary,
        "summary_items": summary_items,
        "notes": notes,
        "keyterms": terms,
        "faithfulness": {
            "sentences": len(every),
            "supported": supported,
            "flagged": len(every) - supported,
            "average": round(sum(x["score"] for x in every) / max(1, len(every)), 2),
            "rewrites_rejected": rejected,
        },
    }