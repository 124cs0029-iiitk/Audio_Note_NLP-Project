import asyncio
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import websockets
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).parent.parent
SR, FRAME = 16000, 1600
URL = "ws://127.0.0.1:8000/ws"
IDLE_SECONDS = 10                  # wait this long after the last message from the server

AUDIO_FILE = ROOT / "replay" / "sample1.mp3"
TIMELINE_FILE = ROOT / "replay" / "sample1_speakers.json"
REFERENCE_MEETING = "sample1"      # clean transcript saved in Step 4
DB_PATH = ROOT / "data" / "meetings.db"


# ---------- ways to damage the audio ----------
def gain_for(audio, timeline, speaker, db):
    out = audio.copy()
    g = 10 ** (db / 20)
    for t in timeline:
        if t["speaker"] == speaker:
            out[int(t["start"] * SR):int(t["end"] * SR)] *= g
    return out


def add_noise(audio, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    speech_rms = float(np.sqrt(np.mean(audio ** 2)))
    noise_rms = speech_rms / (10 ** (snr_db / 20))
    return audio + rng.normal(0, noise_rms, len(audio)).astype(np.float32)


VARIANTS = [
    ("baseline",                     lambda a, tl: a),
    ("Student -12 dB",               lambda a, tl: gain_for(a, tl, "Student", -12)),
    ("Student -24 dB",               lambda a, tl: gain_for(a, tl, "Student", -24)),
    ("everyone -30 dB",              lambda a, tl: a * 10 ** (-30 / 20)),
    ("noise 20 dB",                  lambda a, tl: add_noise(a, 20)),
    ("noise 10 dB",                  lambda a, tl: add_noise(a, 10)),
    ("Student -18 dB + noise 20 dB", lambda a, tl: add_noise(gain_for(a, tl, "Student", -18), 20)),
]


# ---------- metrics ----------
def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def reference_per_turn(timeline):
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT start, end, text FROM segments WHERE meeting_id=? ORDER BY start",
        (REFERENCE_MEETING,)).fetchall()
    db.close()
    refs = [""] * len(timeline)
    for s, e, text in rows:
        mid = (s + e) / 2
        for i, t in enumerate(timeline):
            if t["start"] <= mid < t["end"]:
                refs[i] += " " + text
                break
    return refs


def evaluate(subs, timeline, refs):
    best = []
    for s in subs:
        ov = [overlap(s["start"], s["end"], t["start"], t["end"]) for t in timeline]
        best.append(int(np.argmax(ov)) if ov and max(ov) > 0 else -1)

    ok = sum(1 for s, b in zip(subs, best)
             if b >= 0 and s["speaker"] == timeline[b]["speaker"])
    acc = ok / len(subs) if subs else 0.0

    missed = 0
    for t in timeline:
        covered = sum(overlap(s["start"], s["end"], t["start"], t["end"]) for s in subs)
        if covered < 0.5 * (t["end"] - t["start"]):
            missed += 1

    merged = 0
    for s in subs:
        touched = 0
        for t in timeline:
            o = overlap(s["start"], s["end"], t["start"], t["end"])
            if o >= 0.8 and o >= 0.25 * (t["end"] - t["start"]):
                touched += 1
        if touched >= 2:
            merged += 1

    errors = total = 0
    for i, ref in enumerate(refs):
        hyp = " ".join(s["text"] for s, b in zip(subs, best) if b == i)
        r, h = words(ref), words(hyp)
        errors += edit_distance(r, h)
        total += len(r)
    diff = errors / total if total else 0.0
    return len(subs), missed, merged, acc, diff


# ---------- send one variant to the server ----------
async def run_variant(audio, timeline):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    events = sorted(timeline, key=lambda t: t["start"])
    subs = []
    info = {"id": None, "last": time.time()}

    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:
        async def reader():
            async for raw in ws:
                info["last"] = time.time()
                m = json.loads(raw)
                if m["type"] == "meeting":
                    info["id"] = m["id"]
                elif m["type"] == "subtitle":
                    subs.append(m)

        task = asyncio.create_task(reader())
        idx = 0
        for k in range(len(pcm) // FRAME):
            t = k * FRAME / SR
            while idx < len(events) and events[idx]["start"] <= t:
                await ws.send("speaker:" + events[idx]["speaker"])
                idx += 1
            await ws.send(pcm[k * FRAME:(k + 1) * FRAME].tobytes())

        await ws.send("stop")
        info["last"] = time.time()
        while time.time() - info["last"] < IDLE_SECONDS:
            await asyncio.sleep(0.5)
        task.cancel()
    return subs, info["id"]


def delete_meeting(meeting_id):
    if not meeting_id:
        return
    db = sqlite3.connect(DB_PATH)
    db.execute("DELETE FROM segments WHERE meeting_id=?", (meeting_id,))
    db.commit()
    db.close()


async def main():
    only = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    timeline = json.load(open(TIMELINE_FILE))
    audio = decode_audio(str(AUDIO_FILE), sampling_rate=SR)
    refs = reference_per_turn(timeline)
    print(f"{len(timeline)} true turns, audio {len(audio) / SR:.0f} s\n")
    print(f"{'variant':30} {'found':>6} {'missed':>7} {'merged':>7} {'labels':>7} {'word diff':>10}")

    for name, make in VARIANTS:
        if only and only not in name.lower():
            continue
        variant = make(audio, timeline)
        try:
            subs, mid = await run_variant(variant, timeline)
        except OSError:
            print("Could not connect. Is the server running (uvicorn live.server:app)?")
            return
        delete_meeting(mid)
        found, missed, merged, acc, diff = evaluate(subs, timeline, refs)
        print(f"{name:30} {found:6d} {missed:7d} {merged:7d} {acc:7.0%} {diff:10.0%}", flush=True)
        if only:                                   # show every sentence of the chosen variant
            for s in subs:
                print(f"    {s['start']:5.1f}-{s['end']:5.1f}  {s['speaker']:12} {s['text']}")


asyncio.run(main())