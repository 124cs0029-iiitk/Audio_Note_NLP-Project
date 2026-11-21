import json, sqlite3, sys

meeting_id = sys.argv[1]            # e.g. sample1
timeline_file = sys.argv[2]         # e.g. data/sample1_speakers.json

timeline = json.load(open(timeline_file))

def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

db = sqlite3.connect("data/meetings.db")
rows = db.execute(
    "SELECT rowid, start, end, text FROM segments WHERE meeting_id=?",
    (meeting_id,)).fetchall()

for rowid, start, end, text in rows:
    best = max(timeline, key=lambda t: overlap(start, end, t["start"], t["end"]))
    speaker = best["speaker"] if overlap(start, end, best["start"], best["end"]) > 0 else "Unknown"
    db.execute("UPDATE segments SET speaker=? WHERE rowid=?", (speaker, rowid))
    print(f"{speaker:12} [{start:5.1f}-{end:5.1f}] {text}")

db.commit()
db.close()