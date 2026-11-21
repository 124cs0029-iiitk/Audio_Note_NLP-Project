import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
URL = ("https://raw.githubusercontent.com/Yale-LILY/QMSum/main/"
       "data/Product/jsonl/test.jsonl")
FILE = ROOT / "data" / "qmsum_product_test.jsonl"
DB_PATH = ROOT / "data" / "meetings.db"
REFS = ROOT / "data" / "qmsum_refs.json"

how_many = int(sys.argv[1]) if len(sys.argv) > 1 else 5

if not FILE.exists():
    print("Downloading QMSum (Product / AMI test meetings)...")
    urllib.request.urlretrieve(URL, FILE)
    print(f"Saved {FILE.name}, {FILE.stat().st_size / 1e6:.1f} MB")

meetings = [json.loads(line) for line in open(FILE, encoding="utf-8")]
print(f"{len(meetings)} meetings in the file, using the first {how_many}\n")

db = sqlite3.connect(DB_PATH)
db.execute("""CREATE TABLE IF NOT EXISTS segments(
    meeting_id TEXT, speaker TEXT, start REAL, end REAL, text TEXT)""")

refs = {}
for i, meeting in enumerate(meetings[:how_many], 1):
    meeting_id = f"qmsum_{i}"
    db.execute("DELETE FROM segments WHERE meeting_id=?", (meeting_id,))

    t, n_words, speakers = 0.0, 0, set()
    for turn in meeting["meeting_transcripts"]:
        text = turn["content"].strip()
        if not text:
            continue
        duration = max(1.0, len(text.split()) * 0.4)
        db.execute("INSERT INTO segments VALUES (?,?,?,?,?)",
                   (meeting_id, turn["speaker"], round(t, 2), round(t + duration, 2), text))
        t += duration + 0.3
        n_words += len(text.split())
        speakers.add(turn["speaker"])

    # the human-written summary of the whole meeting
    summary = ""
    queries = meeting.get("general_query_list", [])
    for q in queries:
        if "whole meeting" in str(q.get("query", "")).lower():
            summary = q.get("answer", "")
            break
    if not summary and queries:
        summary = queries[0].get("answer", "")

    refs[meeting_id] = {"summary": summary, "turns": len(meeting["meeting_transcripts"]),
                        "words": n_words, "speakers": sorted(speakers)}
    print(f"{meeting_id}: {len(meeting['meeting_transcripts'])} turns, {n_words} words, "
          f"speakers: {', '.join(sorted(speakers))}")
    print(f"   reference summary ({len(summary.split())} words): {summary[:160]}...\n")

db.commit()
db.close()
json.dump(refs, open(REFS, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"Saved the reference summaries to {REFS.name}")