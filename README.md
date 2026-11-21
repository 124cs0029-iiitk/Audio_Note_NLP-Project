# Audio Note

Live subtitles and structured notes for lectures and Google Meet calls, running **entirely on your own computer**.

Share a Chrome tab (a class video, a talk) or join a Google Meet call, and Audio Note shows subtitles as people speak. When you stop, it builds a report: a summary, timestamped notes, key terms, key dates and, for meetings, action items and **who said what**.

Built as an NLP course project (IIIT Design & Manufacturing, Kurnool) by Poojith Kumar. No cloud service and no LLM API: every model runs locally on the CPU.

## Features

| Mode | What you get |
|---|---|
| **Lecture / video** | Live subtitles from a shared browser tab, then a summary, topic-by-topic notes with timestamps and headings, key terms, key dates, and a faithfulness check that compares every note with the transcript |
| **Google Meet** | Live subtitles with the speaker's name, your own microphone mixed in, then a meeting summary, action items with owners and due dates, key dates, and what each speaker said |
| **History** | Saved reports you can reopen later and export as PDF or DOC |

Other details:

- Speaker names come from **Google Meet itself** (a small Chrome extension reads Meet's "speaking" indicator). There is no voice recognition and no voice profile.
- A review screen lets you fix misheard words or wrong speakers before the report is generated.
- Everything is stored on your machine. Nothing is uploaded.

## How it works

```
Browser tab / microphone
        │  AudioWorklet, 16 kHz PCM, 100 ms chunks
        ▼
   Web app (live/index.html) ──WebSocket──►  FastAPI server (live/server.py)
        ▲                                        │ adaptive voice-activity detection
        │ speaker names                          │ Whisper (faster-whisper, base, int8, CPU)
Chrome extension (extension/)                    │ SQLite
reads Google Meet's speaking indicator           ▼
                                          live subtitles
                                                 │ Stop → review → generate report
                                                 ▼
                              nlp/report.py: BERT NER + BART summariser + rules
```

1. **Capture.** The web app records the shared tab (and the microphone in Meet mode) and streams it to the server.
2. **Speech recognition.** The server cuts the audio into sentences using a loudness threshold that adapts to the background noise, then transcribes each sentence with Whisper in a background queue.
3. **Speaker names (Meet).** The extension tells the app who is speaking and when they started. The server places each speaker change at the right moment in the audio, because the signal for a remote person arrives a little after their voice.
4. **Report.**
   - *Entities:* a BERT NER model plus rules for dates and times.
   - *Meeting summary:* hierarchical BART summarisation (summarise small pieces, then summarise the summaries).
   - *Lecture notes:* the transcript is cleaned, split into topics, and the key sentences of each topic are chosen with TextRank. Every note is checked against the transcript by word overlap and shows where it came from.
   - *Action items and dates:* rule-based extraction.

## Results

All numbers are in [`report/results.md`](report/results.md). The main ones:

**Speech recognition** (100 AMI meeting clips, 494 s of audio, 1,463 reference words, CPU int8)

| Whisper model | Word error rate | Time |
|---|---|---|
| tiny | 19.6% | 101 s |
| **base (used)** | **17.5%** | 218 s |
| small | 15.4% | 772 s |

**Live subtitle delay** (18 sentences streamed in real time): mean 2.44 s, median 2.23 s, 90th percentile 2.94 s.

**Robustness** (a 65 s interview, 18 turns, quiet speakers and added noise): with the adaptive threshold at most 3 of 18 turns were missed even at 10 dB noise. A fixed threshold missed up to 9 turns, or failed completely.

**Meeting summaries** (ROUGE F1 on 5 QMSum meetings)

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| First 100 words | 13.8 | 2.4 | 8.4 |
| Single pass | 30.8 | 8.7 | 17.9 |
| **Hierarchical (used)** | **33.8** | **9.3** | **20.2** |

**Entities** (34 hand-labelled entities in a demo meeting): F1 0.88 strict, 0.97 lenient.

## Quick start (Windows, Chrome)

You need Python 3.11 or newer (developed on 3.13) and Chrome.

```powershell
git clone https://github.com/124cs0029-iiitk/Audio_Note_NLP-Project.git
cd Audio_Note_NLP-Project

python -m venv .venv
.venv\Scripts\activate

pip install fastapi "uvicorn[standard]" faster-whisper numpy
pip install --only-binary=:all: "av>=14,<17"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

Start the server from the project folder:

```powershell
uvicorn live.server:app
```

Wait for `Application startup complete`, then open <http://127.0.0.1:8000>.

The first start downloads the models (about 2 GB in total: Whisper base, a BERT NER model and a BART summariser). After that, everything works offline.

## Use it

**Lecture / video**

1. Open the video in another Chrome tab.
2. In Audio Note, open **Lecture / video** and press **Start recording**.
3. Choose the video tab and tick **Also share tab audio**.
4. Press **Stop**, then **Review & generate report**.

**Google Meet**

1. Install the extension (next section), then join your Meet call in a **new tab**.
2. In Audio Note, open **Google Meet**. The participants appear within a few seconds.
3. Keep your microphone **unmuted in Meet** (that is how Meet knows who is speaking) and wear headphones, so the other person's voice is not recorded twice.
4. Press **Start recording**, choose the Meet tab, tick **Also share tab audio**, and allow the microphone.
5. At the end press **Stop**, review the transcript, and generate the report.

## Install the Chrome extension

1. Open `chrome://extensions` and switch on **Developer mode**.
2. Click **Load unpacked** and choose the `extension` folder of this repo.
3. After loading or reloading the extension, **open the Meet tab again in a new tab**. Chrome only connects the extension to pages that were opened after it was loaded.

The extension reads only the participants' names and the speaking indicator on `meet.google.com`, and passes them to the Audio Note page on `127.0.0.1:8000`. It sends nothing anywhere else.

## Privacy and consent

- Audio, transcripts and reports stay on your computer.
- The sign-in in the web app is a **local profile stored in your browser**. It separates histories on a shared computer, but it is not real authentication.
- If you transcribe a call, tell the other participants. Recording other people may need their consent where you live.

## Limitations

- **Names and technical terms** are often misheard by Whisper base (for example, "NLP" can become "handle"). Use the review screen before generating a report.
- **Speaker names depend on Google Meet's speaking indicator.** Google can change the page structure, which would break the extension. Meet only marks you as speaking while your microphone is unmuted. In informal two-device tests most sentences got the right name, and the first words of a turn were the usual mistakes.
- The summariser is trained on short chat conversations, so lecture summaries are built from selected transcript sentences instead of free rewriting, and the faithfulness check only compares words. It catches invented content but not every overclaim.
- CPU only. A one-hour lecture takes about one hour to transcribe, because it runs in real time.
- Tested on Windows with Chrome.

## Project structure

```
live/        FastAPI server (server.py) and the web app (index.html)
extension/   Chrome extension: reads speaker names from Google Meet
nlp/         Report builder (report.py), lecture notes (lecture.py),
             offline transcription and speaker tools
eval/        Evaluation scripts: WER, ROUGE, NER, latency, noise tests
report/      results.md with all evaluation numbers
replay/      Speaker timelines for test recordings (the recordings are not included)
```

## Models and data

- Speech recognition: [Whisper](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Entities: [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)
- Summarisation: [`philschmid/bart-large-cnn-samsum`](https://huggingface.co/philschmid/bart-large-cnn-samsum)
- Evaluation data: AMI meeting corpus and QMSum

## Roadmap

- Optional hint-words box to improve names and technical terms
- Speaker attribution that also uses which audio source is active (your microphone or the tab)
- A stricter faithfulness check based on entailment