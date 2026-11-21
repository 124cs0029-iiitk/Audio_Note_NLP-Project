import sys, time
from faster_whisper import WhisperModel

audio = sys.argv[1]                  # e.g. data/sample1.mp3
duration = float(sys.argv[2])        # length of the audio in seconds

for size in ["tiny", "base", "small"]:
    model = WhisperModel(size, device="cpu", compute_type="int8")
    t0 = time.time()
    segments, _ = model.transcribe(audio, vad_filter=True, language="en")
    text = " ".join(s.text.strip() for s in segments)
    took = time.time() - t0
    print(f"{size:6} time={took:5.1f}s  speed={duration/took:4.1f}x real-time  words={len(text.split())}")