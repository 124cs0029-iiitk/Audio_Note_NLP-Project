# Audio Note

Live subtitles and structured notes for lectures and Google Meet calls. Everything runs on your own computer: no cloud service and no LLM API.

Share a Chrome tab (a class video, a talk) or join a Google Meet call, and Audio Note shows subtitles while people speak. When you stop, it builds a report: a summary, notes, key terms, key dates and, for meetings, action items and who said what.

Built as an NLP course project (IIIT Design & Manufacturing, Kurnool) by Poojith Kumar.

## Features

| Section | What you get |
|---|---|
| **Lecture / video** | Live subtitles from a shared browser tab, then a summary, notes grouped by topic, key terms, key dates, and a faithfulness check that compares every summary sentence with the transcript |
| **Google Meet** | Live subtitles with each speaker's name and your own microphone, then a meeting summary, action items with owners and due dates, key dates, and what each speaker said |
| **History** | Every saved session. Reopen its report, export it as PDF or DOC, regenerate it, or delete it |

Other details:

- Speaker names come from **Google Meet itself**. A small Chrome extension reads Meet's "speaking" indicator. There is no voice recognition and no voice profile.
- A review screen lets you fix misheard words or wrong speakers before the report is generated.
- Everything is stored on your computer. Nothing is uploaded.

## How it works

```
Browser tab / microphone
        |  AudioWorklet: 16 kHz audio, 100 ms chunks
        v
  Web app (live/index.html) --WebSocket--> FastAPI server (live/server.py)
        ^                                       | adaptive voice-activity detection
        | speaker names                         | Whisper (faster-whisper, base, int8, CPU)
Chrome extension (extension/)                   | SQLite (data/meetings.db)
reads Google Meet's speaking indicator          v
                                          live subtitles
                                                | Stop -> review -> generate report
                                                v
                             nlp/report.py   BERT NER + BART summariser + rules
                             nlp/lecture.py  TF-IDF topics + TextRank + faithfulness check
```

1. **Capture.** The page records the shared tab (and the microphone in Meet mode) and streams it to the server.
2. **Speech recognition.** The server cuts the audio into sentences with a loudness threshold that adapts to background noise, then transcribes each sentence with Whisper in a background queue. Silence phrases and repeated words that Whisper invents are filtered out.
3. **Speaker names (Meet).** The extension reports who is speaking and when they started. The server places each change at the right moment in the audio, because the signal for a remote person arrives a little late.
4. **Meeting report.** A BERT model finds names, places and organisations. Dates, tasks and requests come from rules. A BART model summarises in pieces and then summarises the summaries.
5. **Lecture notes.** The transcript is cleaned and split into topics with TF-IDF and cosine similarity. TextRank picks the key sentences of each topic. Every note is checked against the transcript.

## Technologies

| Technology | Used for |
|---|---|
| Python 3.13, FastAPI, Uvicorn | The backend, the REST routes and the live WebSocket |
| faster-whisper (Whisper `base`, int8, CPU) | Speech recognition |
| SQLite | Storing the transcript |
| NumPy | Loudness, TF-IDF, TextRank |
| PyTorch, Hugging Face Transformers | `dslim/bert-base-NER` (entities) and `philschmid/bart-large-cnn-samsum` (summaries) |
| Regular expressions | Dates, tasks, requests |
| HTML, CSS, JavaScript | The web app, with no frameworks. It uses the Web Audio API (AudioWorklet), screen capture, WebSocket and localStorage |
| Chrome extension (Manifest V3) | Reading speaker names from Google Meet |

## Quick start (Windows, Chrome)

You need Python 3.11 or newer (developed on 3.13) and Chrome.

```powershell
git clone https://github.com/124cs0029-iiitk/Audio_Note_NLP-Project.git
cd Audio_Note_NLP-Project
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn live.server:app
```

Wait for `Application startup complete`, then open <http://127.0.0.1:8000>.

The first start downloads the models (about 2 GB in total). After that, everything works offline.

## Use it

**Lecture / video**

1. Open the video in another Chrome tab.
2. In Audio Note, open **Lecture / video** and press **Start recording**.
3. Choose the video tab and tick **Also share tab audio**.
4. Press **Stop**, then **Review & generate report**.

**Google Meet**

1. Install the extension (next section) and join your Meet call in a **new tab**.
2. In Audio Note, open **Google Meet**. The participants appear within a few seconds.
3. Keep your microphone **unmuted in Meet** (that is how Meet knows who is speaking) and wear headphones.
4. Press **Start recording**, choose the Meet tab, tick **Also share tab audio**, and allow the microphone.
5. At the end press **Stop**, review the transcript, and generate the report.

**History** keeps every saved session. You can open it, export it as PDF or DOC, or delete it. Deleting removes the report and the transcript of that session and cannot be undone.

## Install the Chrome extension

1. Open `chrome://extensions` and switch on **Developer mode**.
2. Click **Load unpacked** and choose the `extension` folder.
3. After loading or reloading it, open the Meet tab again in a new tab. Chrome only connects the extension to pages opened after it was loaded.

The extension reads only the participants' names and the speaking indicator on `meet.google.com`, and passes them to the Audio Note page on `127.0.0.1:8000`. It sends nothing anywhere else.

## Project structure

```
live/        server.py (backend) and index.html (the web app)
nlp/         report.py (meeting reports) and lecture.py (lecture notes)
extension/   Chrome extension that reads speaker names from Google Meet
requirements.txt
```

`data/` is created on the first recording and holds `meetings.db` and the saved reports. It is not part of the repository.

## Privacy

- Audio, transcripts and reports stay on your computer.
- The sign-in is a **local profile in your browser**. It keeps histories apart on a shared computer, but it is not real authentication.
- If you transcribe a call, tell the other participants. Recording people may need their consent where you live.

## Limitations

- Whisper `base` often mishears names and technical terms. Use the review screen before generating a report.
- Speaker names depend on Google Meet's speaking indicator. Meet only marks you as speaking while your microphone is unmuted. If Google changes the page, the extension may stop working.
- The faithfulness check compares words. It catches invented content, not every wrong combination of true facts.
- CPU only. A one-hour lecture takes about one hour to transcribe, because it runs in real time.
- Tested on Windows with Chrome.

## Evaluation (measured during development with separate scripts)

| Measure | Result |
|---|---|
| Whisper word error rate, 100 AMI meeting clips | tiny 19.6%, **base 17.5%**, small 15.4% |
| Live subtitle delay (18 sentences) | mean 2.44 s, median 2.23 s |
| Meeting summaries, ROUGE-1 on 5 QMSum meetings | **hierarchical 33.8**, single pass 30.8, first 100 words 13.8 |
| Entities, 34 hand-labelled entities | F1 0.88 strict, 0.97 lenient |