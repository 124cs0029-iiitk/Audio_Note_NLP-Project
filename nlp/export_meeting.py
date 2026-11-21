import json, sqlite3, sys

meeting_id = sys.argv[1]                       # e.g. sample1
out_file = sys.argv[2]                         # e.g. data/sample1.json

db = sqlite3.connect("data/meetings.db")
rows = db.execute(
    "SELECT speaker, start, end, text FROM segments "
    "WHERE meeting_id=? ORDER BY start", (meeting_id,)).fetchall()
db.close()

data = {
    "meeting_id": meeting_id,
    "segments": [
        {"speaker": s, "start": st, "end": en, "text": t}
        for s, st, en, t in rows
    ],
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Exported {len(rows)} segments to {out_file}")