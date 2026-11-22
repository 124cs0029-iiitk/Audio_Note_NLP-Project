import asyncio
import importlib.util
import json
import re
import sqlite3
import threading
import time
import wave
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()
ROOT = Path(__file__).parent.parent
PAGE = Path(__file__).parent / "index.html"
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "meetings.db"

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Whisper base model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model ready")

SR = 16000
FRAME = 1600                  # 100 ms of audio
SILENCE_FRAMES = 8            # 0.8 s of silence ends a sentence
CHANGE_SILENCE_FRAMES = 4     # after a speaker change, a 0.4 s pause is enough
MIN_SPEECH_FRAMES = 3         # ignore clicks shorter than 0.3 s
MAX_UTT_FRAMES = 150          # cut very long sentences at 15 s
CUT_AT_SPEAKER_CHANGE = True  # end the sentence soon after the speaker changes
SAVE_DEBUG_WAV = False        # set True to save each sentence in data/
SPEAKER_LEAD = 0.2            # seconds: the speaking indicator reacts a little after the voice starts
SPEAKER_WAIT = 3.0            # seconds to wait after a sentence began for Meet to say who started it
USE_NAME_HINTS = False        # True: tell Whisper the participants' names (it may then write the names on quiet audio)
BEAM_SIZE = 1                 # 1 = faster live subtitles, 5 = a little more accurate (Whisper's default)
COLLAPSE_REPEATS = True       # "Good morning, good morning!" -> "Good morning!"  (False: keep every repeat)

# Words Whisper often gets wrong (names, college, tools). Add your own here.
# They are given to Whisper as a hint, together with the names of the Meet participants.
HINT_TERMS = []

# adaptive speech threshold (based on the background-noise level)
HIST_FRAMES = 200             # look at the last 20 s
WARMUP_FRAMES = 20            # first 2 s: use a fixed threshold
WARMUP_THRESH = 0.001
FLOOR_PERCENTILE = 15         # the quietest 15% of frames = background level
NOISE_FACTOR = 3.0            # speech must be 3x louder than the background
MIN_THRESH = 0.0005
MAX_THRESH = 0.03


# ---------- report module (nlp/report.py), loaded once ----------
_report_mod = None
_report_lock = threading.Lock()


def get_report_module():
    global _report_mod
    with _report_lock:
        if _report_mod is None:
            spec = importlib.util.spec_from_file_location(
                "meeting_report", ROOT / "nlp" / "report.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._load()                 # load NER + summarization models
            _report_mod = mod
        return _report_mod


def _warm_up():
    try:
        get_report_module()
        print("Report models ready")
    except Exception as e:
        print("Report models could not be preloaded:", e)


print("Preloading report models in the background...")
threading.Thread(target=_warm_up, daemon=True).start()


# ---------- helpers ----------
def has_words(text):
    return bool(re.search(r"[A-Za-z0-9]", text))


def normalize(audio):
    """Make quiet audio louder before Whisper (up to 20x)."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return audio
    return audio * min(0.9 / peak, 20.0)


def make_prompt(names):
    """Hint text for Whisper, or None. The participants' names are only used if USE_NAME_HINTS is
    True: on quiet or noisy audio Whisper then tends to write the names instead of the speech."""
    terms = []
    for t in (list(names) if USE_NAME_HINTS else []) + HINT_TERMS:
        if t and t not in terms:
            terms.append(t)
    return ("Vocabulary: " + ", ".join(terms) + ".") if terms else None


# things Whisper writes when there is only silence or noise
HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "we'll see you in the next one",
    "see you in the next video", "please subscribe", "subtitles by the amara.org community",
)


def is_hallucination(text):
    """True for a repeated word ("Kumar! Kumar! Kumar!") or a typical silence phrase."""
    t = " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())
    if not t:
        return True
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return any(t == h or (t.startswith(h) and len(t) <= len(h) + 5) for h in HALLUCINATIONS)


_REPEAT_RE = re.compile(r"\b((?:[\w']+\s+){0,3}[\w']+)([,.!?]*\s+)\1\b([,.!?]*)", re.I)


def collapse_repeats(text):
    """Whisper often says a phrase twice: 'Good morning, good morning!' -> 'Good morning!'"""
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_RE.sub(lambda m: m.group(1) + m.group(3), text)
    return text


def run_whisper(audio, vad, prompt):
    segments, _ = model.transcribe(
        audio, language="en",
        vad_filter=vad,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        beam_size=BEAM_SIZE,
    )
    kept = []
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > 0.6 and logprob < -1.0:
            continue                        # Whisper itself says that this was not speech
        kept.append(seg.text.strip())
    return " ".join(kept).strip()


def transcribe(audio, names=()):
    audio = normalize(audio)
    prompt = make_prompt(names)
    text = run_whisper(audio, True, prompt)
    if not has_words(text):
        text = run_whisper(audio, False, prompt)
    if not has_words(text) or is_hallucination(text):
        return ""
    return collapse_repeats(text) if COLLAPSE_REPEATS else text


def threshold(hist):
    """Loudness level that counts as speech, based on the recent background level."""
    if len(hist) < WARMUP_FRAMES:
        return WARMUP_THRESH
    floor = float(np.percentile(hist, FLOOR_PERCENTILE))
    return min(max(NOISE_FACTOR * floor, MIN_THRESH), MAX_THRESH)


def save_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def save_segment(meeting_id, speaker, start, end, text):
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS segments(
        meeting_id TEXT, speaker TEXT, start REAL, end REAL, text TEXT)""")
    cur = db.execute("INSERT INTO segments VALUES (?,?,?,?,?)",
                     (meeting_id, speaker, start, end, text))
    row_id = cur.lastrowid
    db.commit()
    db.close()
    return row_id


def apply_corrections(meeting_id, items):
    if not items:
        return
    db = sqlite3.connect(DB_PATH)
    for it in items:
        if "id" in it and "speaker" in it and "text" in it:
            db.execute(
                "UPDATE segments SET speaker=?, text=? WHERE rowid=? AND meeting_id=?",
                (it["speaker"], it["text"], it["id"], meeting_id))
    db.commit()
    db.close()


def _safe_id(meeting_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", meeting_id)


REPORT_MODES = ("meeting", "lecture", "general", "class")


def _report_file(meeting_id: str, mode: str | None = None) -> Path | None:
    """The saved report of exactly this session (a session whose id starts the same way is never used)."""
    sid = _safe_id(meeting_id)
    if mode:
        p = REPORTS_DIR / f"{sid}_{_safe_id(mode)}.json"
        if p.exists():
            return p
    candidates = [REPORTS_DIR / f"{sid}_{m}.json" for m in REPORT_MODES] + [REPORTS_DIR / f"{sid}.json"]
    existing = [p for p in candidates if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def save_report_file(meeting_id: str, mode: str, payload: dict):
    sid = _safe_id(meeting_id)
    target = REPORTS_DIR / f"{sid}_{_safe_id(mode)}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_speaker(events, a, b, fallback):
    """events = [(time, name), ...] sorted by time.
    Return the speaker who was active for most of the span a..b (seconds).
    "Unknown" only means that nobody has been reported yet, so a real name wins over it
    as soon as that person was reported during at least 0.3 s of the sentence."""
    total = defaultdict(float)
    for i, (t0, name) in enumerate(events):
        t1 = events[i + 1][0] if i + 1 < len(events) else float("inf")
        overlap = min(b, t1) - max(a, t0)
        if overlap > 0:
            total[name] += overlap
    if not total:
        return fallback
    known = {n: v for n, v in total.items() if n != "Unknown" and v >= 0.3}
    if known:
        return max(known, key=known.get)
    return max(total, key=total.get)


# ---------- routes ----------
@app.get("/")
def index():
    return HTMLResponse(
        PAGE.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/report/{meeting_id}")
def get_saved_report(meeting_id: str, mode: str | None = None):
    path = _report_file(meeting_id, mode)
    if not path:
        return JSONResponse({"found": False, "segments": []}, status_code=200)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["found"] = True
    return JSONResponse(data)


@app.delete("/api/report/{meeting_id}")
def delete_session(meeting_id: str):
    """Delete one session: its saved report files and its transcript lines.
    Only the exact files of this session are removed (another session whose id starts the same way is safe)."""
    sid = _safe_id(meeting_id)
    names = [f"{sid}.json"] + [f"{sid}_{mode}.json" for mode in REPORT_MODES]
    reports = 0
    for name in names:
        path = REPORTS_DIR / name
        if path.exists():
            path.unlink()
            reports += 1
    lines = 0
    if DB_PATH.exists():
        db = sqlite3.connect(DB_PATH)
        try:
            lines = db.execute("DELETE FROM segments WHERE meeting_id=?", (meeting_id,)).rowcount
            db.commit()
        except sqlite3.OperationalError:        # no table yet: nothing to delete
            lines = 0
        finally:
            db.close()
    return {"deleted": True, "reports": reports, "lines": lines}


@app.post("/api/report/{meeting_id}")
async def make_report(meeting_id: str, mode: str = "meeting", body: dict = Body(default={})):
    corrections = body.get("segments", []) if isinstance(body, dict) else []
    apply_corrections(meeting_id, corrections)

    def work():
        rep = get_report_module()
        segs = rep.load_segments(meeting_id)
        report = rep.build_report(segs, mode)
        if segs:
            save_report_file(meeting_id, mode, report)
        return report

    return await asyncio.to_thread(work)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    meeting_id = datetime.now().strftime("live_%Y%m%d_%H%M%S")
    current_speaker = "Unknown"
    speaker_events = [(0.0, "Unknown")]     # when each speaker change happened (audio time)
    hint_names = []                         # names seen in this meeting, used as Whisper hints
    speaker_arrivals = []                   # (when the message arrived, name): what Meet reported, and when
    stopped = False                         # True after "stop": no more audio will arrive

    pending = np.zeros(0, dtype=np.float32)
    preroll = deque(maxlen=2)               # keeps 0.2 s before speech starts
    rms_hist = deque(maxlen=HIST_FRAMES)    # recent loudness values
    utt, in_speech, silence, speech_frames = [], False, 0, 0
    silence_needed = SILENCE_FRAMES
    frame_idx, utt_start = 0, 0
    jobs = asyncio.Queue()                  # finished sentences waiting for Whisper

    def finish():
        """Close the current sentence and hand it to the transcription worker."""
        nonlocal utt, in_speech, silence, silence_needed
        if speech_frames >= MIN_SPEECH_FRAMES:
            start = utt_start * FRAME / SR
            end = start + max(len(utt) - silence, 1) * FRAME / SR   # without trailing silence
            speaker = pick_speaker(speaker_events, start, end, current_speaker)
            jobs.put_nowait((speaker, start, end, np.concatenate(utt)))
        utt, in_speech, silence = [], False, 0
        silence_needed = SILENCE_FRAMES

    def stream_now():
        """How many seconds of audio have arrived so far."""
        return (frame_idx * FRAME + len(pending)) / SR

    def speaker_for(a, b, fallback):
        """The speaker that Google Meet reported shortly after this sentence began. The signal
        for a remote person comes a little after their voice, so it is looked for in the next
        few seconds. If there is none, the person who was active for most of the sentence."""
        for arrived, name in speaker_arrivals:
            if a <= arrived <= a + SPEAKER_WAIT and name != "Unknown":
                return name
        return pick_speaker(speaker_events, a, b, fallback)

    emit_tasks = []                         # sentences that are waiting to be shown, in order

    async def emit(prev, speaker, start, end, text):
        """Show one sentence. It waits for the one before it, so the order never changes."""
        if prev is not None:
            await asyncio.wait([prev])
        try:
            # with more than one person, wait until Meet has had time to say who started
            if len(hint_names) >= 2 or speaker == "Unknown":
                while not stopped and stream_now() < start + SPEAKER_WAIT:
                    await asyncio.sleep(0.1)
                speaker = speaker_for(start, end, speaker)
            start, end = round(start, 2), round(end, 2)
            row_id = save_segment(meeting_id, speaker, start, end, text)
            print(f"[{meeting_id}] {speaker}: {text}")
            await websocket.send_json({
                "type": "subtitle", "id": row_id, "speaker": speaker,
                "text": text, "start": start, "end": end,
            })
        except (WebSocketDisconnect, RuntimeError):
            return                          # the browser disconnected
        except Exception as e:
            print("Could not show a subtitle:", repr(e))

    async def worker():
        n = 0
        while True:
            job = await jobs.get()
            if job is None:                 # no more sentences are coming
                await asyncio.gather(*emit_tasks, return_exceptions=True)
                try:
                    await websocket.send_json({"type": "done"})
                except Exception:
                    pass
                return
            speaker, start, end, audio = job
            n += 1
            try:
                if SAVE_DEBUG_WAV:
                    save_wav(DATA_DIR / f"debug_utt{n}.wav", audio)
                await websocket.send_json({"type": "status", "text": "transcribing..."})
                t0 = time.time()
                text = await asyncio.to_thread(transcribe, audio, list(hint_names))
                took = time.time() - t0
                print(f"  sentence {n}: {end - start:.1f} s of audio transcribed in {took:.1f} s "
                      f"({jobs.qsize()} waiting)" + ("" if text else "  -> nothing kept (no speech)"))
                if text:
                    prev = emit_tasks[-1] if emit_tasks else None
                    emit_tasks.append(asyncio.create_task(emit(prev, speaker, start, end, text)))
                state = "transcribing..." if not jobs.empty() else "listening..."
                await websocket.send_json({"type": "status", "text": state})
            except (WebSocketDisconnect, RuntimeError):
                return                      # the browser disconnected
            except Exception as e:
                print("Transcription failed:", repr(e))

    worker_task = asyncio.create_task(worker())

    try:
        await websocket.send_json({"type": "meeting", "id": meeting_id})
        await websocket.send_json({"type": "status", "text": "connected"})
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            txt = msg.get("text")
            if txt == "stop":
                stopped = True
                if in_speech:
                    finish()
                jobs.put_nowait(None)       # tells the worker to send "done" at the end
                continue
            if txt and txt.startswith("speaker:"):
                name, _, ago = txt[len("speaker:"):].partition("|")   # "Name|ms": speech began ms ago
                name = name.strip() or "Unknown"
                try:
                    lead = float(ago) / 1000.0 if ago else 0.0
                except ValueError:
                    lead = 0.0
                repeat = lead < 0               # a negative number: the same person is still speaking
                lead = abs(lead)
                if name == current_speaker and not ago:
                    continue            # the page repeats the current speaker every few seconds: no change
                if CUT_AT_SPEAKER_CHANGE and in_speech and name != current_speaker:
                    if silence >= CHANGE_SILENCE_FRAMES:
                        finish()            # the pause already happened
                    else:
                        silence_needed = CHANGE_SILENCE_FRAMES   # end at the next pause
                current_speaker = name
                if name != "Unknown" and name not in hint_names:
                    hint_names.append(name)
                if not repeat:                  # a repeat may be late, so it never decides who began a sentence
                    speaker_arrivals.append((stream_now(), name))
                # position in the audio stream where this change arrived
                t = (frame_idx * FRAME + len(pending)) / SR - lead - SPEAKER_LEAD
                t = max(t, speaker_events[-1][0] + 0.01)           # keep the events in order
                if abs(speaker_events[-1][0] - t) < 1e-6:
                    speaker_events[-1] = (t, name)
                else:
                    speaker_events.append((t, name))
                continue
            if not msg.get("bytes"):
                continue

            samples = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
            pending = np.concatenate([pending, samples])

            while len(pending) >= FRAME:
                frame, pending = pending[:FRAME], pending[FRAME:]
                rms = float(np.sqrt(np.mean(frame ** 2)))
                loud = rms > threshold(rms_hist)
                rms_hist.append(rms)

                if loud:
                    if not in_speech:
                        in_speech = True
                        utt = list(preroll)
                        utt_start = frame_idx - len(utt)
                        preroll.clear()
                        speech_frames = 0
                    utt.append(frame)
                    speech_frames += 1
                    silence = 0
                elif in_speech:
                    utt.append(frame)
                    silence += 1
                else:
                    preroll.append(frame)
                frame_idx += 1

                if in_speech and (silence >= silence_needed or len(utt) >= MAX_UTT_FRAMES):
                    finish()
    except WebSocketDisconnect:
        pass
    finally:
        worker_task.cancel()
        for task in emit_tasks:
            task.cancel()