import sqlite3, sys
from faster_whisper import WhisperModel

audio = sys.argv[1]                 # e.g. data/sample1.mp3
meeting_id = sys.argv[2] if len(sys.argv) > 2 else "sample1"

print("Loading model (first run downloads it, please wait)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, vad_filter=True, language="en")

db = sqlite3.connect("data/meetings.db")
db.execute("""CREATE TABLE IF NOT EXISTS segments(
    meeting_id TEXT, speaker TEXT, start REAL, end REAL, text TEXT)""")

for s in segments:
    text = s.text.strip()
    print(f"[{s.start:6.1f} - {s.end:6.1f}] {text}")
    db.execute("INSERT INTO segments VALUES (?,?,?,?,?)",
               (meeting_id, "Unknown", s.start, s.end, text))

db.commit()
db.close()
print("Saved to data/meetings.db")