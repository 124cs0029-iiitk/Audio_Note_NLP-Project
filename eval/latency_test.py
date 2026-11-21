import asyncio
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import websockets
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).parent.parent
SR, FRAME = 16000, 1600
URL = "ws://127.0.0.1:8000/ws"
AUDIO_FILE = ROOT / "replay" / "sample1.mp3"
TIMELINE_FILE = ROOT / "replay" / "sample1_speakers.json"
DB_PATH = ROOT / "data" / "meetings.db"


async def main():
    timeline = json.load(open(TIMELINE_FILE))
    events = sorted(timeline, key=lambda t: t["start"])
    audio = decode_audio(str(AUDIO_FILE), sampling_rate=SR)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    n_chunks = len(pcm) // FRAME

    arrivals = []                     # (time the subtitle arrived, the subtitle)
    info = {"id": None}
    done = asyncio.Event()

    try:
        connection = await websockets.connect(URL, max_size=None, ping_interval=None)
    except OSError:
        print("Could not connect. Is the server running (uvicorn live.server:app)?")
        return

    async with connection as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "meeting":
                    info["id"] = m["id"]
                elif m["type"] == "subtitle":
                    arrivals.append((time.time(), m))
                elif m["type"] == "done":
                    done.set()

        task = asyncio.create_task(reader())
        print(f"Playing {n_chunks / 10:.0f} s of audio at normal speed...", flush=True)

        t0 = time.time()                          # stream time 0
        idx = 0
        for k in range(n_chunks):
            wait = t0 + k * FRAME / SR - time.time()
            if wait > 0:
                await asyncio.sleep(wait)         # keep real-time speed
            t = k * FRAME / SR
            while idx < len(events) and events[idx]["start"] <= t:
                await ws.send("speaker:" + events[idx]["speaker"])
                idx += 1
            await ws.send(pcm[k * FRAME:(k + 1) * FRAME].tobytes())

        await ws.send("stop")
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            print("(the server did not send 'done' within 120 s)")
        task.cancel()

    if info["id"]:
        db = sqlite3.connect(DB_PATH)
        db.execute("DELETE FROM segments WHERE meeting_id=?", (info["id"],))
        db.commit()
        db.close()

    # delay = arrival time minus the moment the speech of that sentence ended
    delays = [arrival - (t0 + m["end"]) for arrival, m in arrivals]
    print(f"\n{len(delays)} subtitles\n")
    print(f"{'speech ended':>12} {'delay':>7}  speaker       text")
    for (arrival, m), d in zip(arrivals, delays):
        print(f"{m['end']:11.1f}s {d:6.2f}s  {m['speaker']:12}  {m['text'][:38]}")

    print(f"\nmean    {np.mean(delays):.2f} s")
    print(f"median  {np.median(delays):.2f} s")
    print(f"90th %  {np.percentile(delays, 90):.2f} s")
    print(f"max     {np.max(delays):.2f} s")


asyncio.run(main())