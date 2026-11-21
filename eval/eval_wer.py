import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).parent.parent
CLIP_DIR = ROOT / "data" / "ami_clips"
CLIP_FILE = ROOT / "data" / "ami_clips.json"
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SIZES = ["tiny", "base", "small"]
SR = 16000


# ---------- 1. get the clips (once) ----------
def download_clips():
    from datasets import Audio, load_dataset
    print(f"Streaming AMI and selecting {N_CLIPS} clips (this downloads about 100 MB)...",
          flush=True)
    ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=0, buffer_size=1000)

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    for ex in ds:
        duration = ex["end_time"] - ex["begin_time"]
        if not (2.0 <= duration <= 15.0 and len(ex["text"].split()) >= 6):
            continue
        name = f"clip{len(clips):03d}.wav"
        (CLIP_DIR / name).write_bytes(ex["audio"]["bytes"])
        clips.append({"file": name, "ref": ex["text"], "duration": round(duration, 2),
                      "meeting": ex["meeting_id"], "speaker": ex["speaker_id"]})
        if len(clips) % 20 == 0:
            print(f"  {len(clips)} clips", flush=True)
        if len(clips) >= N_CLIPS:
            break
    json.dump(clips, open(CLIP_FILE, "w", encoding="utf-8"), indent=1)
    return clips


if CLIP_FILE.exists():
    clips = json.load(open(CLIP_FILE, encoding="utf-8"))
else:
    clips = download_clips()
meetings = sorted({c["meeting"] for c in clips})
total_audio = sum(c["duration"] for c in clips)
print(f"{len(clips)} clips, {total_audio:.0f} s of audio, meetings: {', '.join(meetings)}\n")


# ---------- 2. text cleaning, the same for reference and hypothesis ----------
FILLERS = {"uh", "um", "mm", "hmm", "hm", "mhm", "mmhmm", "uhhuh", "erm", "er", "huh"}
UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    if n < 20:
        return UNITS[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + UNITS[n % 10])
    if n < 1000:
        return UNITS[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    return str(n)


def norm(text):
    text = text.lower().replace("-", " ")
    text = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", text)
    text = re.sub(r"[^a-z' ]", " ", text)
    words = [w.strip("'") for w in text.split()]
    return [w for w in words if w and w not in FILLERS]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ---------- 3. run Whisper like the live server does ----------
def normalize_audio(audio):
    peak = float(np.max(np.abs(audio)))
    return audio if peak < 1e-4 else audio * min(0.9 / peak, 20.0)


def recognize(model, audio):
    audio = normalize_audio(audio)
    for vad in (True, False):                   # same fallback as the server
        segments, _ = model.transcribe(audio, language="en", vad_filter=vad,
                                       condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        if re.search(r"[A-Za-z0-9]", text):
            return text
    return ""


results = {}
for size in SIZES:
    cache_file = ROOT / "data" / f"ami_hyp_{size}.json"
    if cache_file.exists():
        results[size] = json.load(open(cache_file, encoding="utf-8"))
        if len(results[size]["hyp"]) == len(clips):
            print(f"{size}: loaded from {cache_file.name}", flush=True)
            continue

    print(f"{size}: loading the model...", flush=True)
    model = WhisperModel(size, device="cpu", compute_type="int8")
    hyp, t0 = [], time.time()
    for i, c in enumerate(clips):
        audio = decode_audio(str(CLIP_DIR / c["file"]), sampling_rate=SR)
        hyp.append(recognize(model, audio))
        if (i + 1) % 20 == 0:
            print(f"  {size}: {i + 1}/{len(clips)} clips, {time.time() - t0:.0f} s", flush=True)
    results[size] = {"hyp": hyp, "seconds": round(time.time() - t0, 1)}
    json.dump(results[size], open(cache_file, "w", encoding="utf-8"), indent=1)


# ---------- 4. scores ----------
print(f"\n{'model':7} {'WER':>7} {'errors':>7} {'ref words':>9} {'time s':>7} {'speed':>7}")
for size in SIZES:
    errors = words = 0
    for c, h in zip(clips, results[size]["hyp"]):
        ref = norm(c["ref"])
        errors += edit_distance(ref, norm(h))
        words += len(ref)
    secs = results[size]["seconds"]
    print(f"{size:7} {100 * errors / words:6.1f}% {errors:7d} {words:9d} {secs:7.0f} "
          f"{total_audio / secs:6.1f}x")

print("\nExamples (reference / base):")
for c, h in list(zip(clips, results["base"]["hyp"]))[:5]:
    print("  REF :", " ".join(norm(c["ref"])))
    print("  BASE:", " ".join(norm(h)), "\n")