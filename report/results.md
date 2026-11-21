# Results (raw material for the report)

All timings are on a laptop CPU (int8). The Whisper models are faster-whisper `tiny`, `base`
and `small`.

## 1. Speech recognition speed (one 65 s recording)

| Model | Time | Speed (times real time) |
|---|---|---|
| tiny | 11.0 s | 5.9x |
| base | 16.8 s | 3.9x |
| small | 56.2 s | 1.2x |

## 2. Speech recognition accuracy on real meeting speech (AMI)

Data: AMI Meeting Corpus, headset microphone ("ihm"), test split, Hugging Face
`edinburghcstr/ami`. 100 random clips of 2 to 15 s with at least 6 words, from meetings
EN2002a, EN2002c, ES2004c, IS1009d (494 s of audio, 1,463 reference words).
Text cleaning for both reference and output: lowercase, no punctuation, numbers below 1000
spelled out, filler sounds (uh, um, mm, hmm) removed. Server settings: loudness boost,
speech filter with fallback.

| Model | WER | Errors | Time | Speed |
|---|---|---|---|---|
| tiny | 19.6% | 287 | 101 s | 4.9x |
| base | 17.5% | 256 | 218 s | 2.3x |
| small | 15.4% | 225 | 772 s | 0.6x |

Notes: spelling variants (colours/colors) and "gonna"/"going to" count as errors, so the true
WER is somewhat lower. Whisper sometimes adds words from cross-talk. Only 100 clips, so a
2-point difference is not firm. `base` is used for live subtitles (small is slower than real
time).

## 3. Live pipeline robustness (stress test)

Data: one 65 s interview recording, 18 true speaker turns, speaker names supplied by a
timeline file. Damaged copies are streamed to the running server.

| Variant | First version (fixed loudness threshold) | Final version (adaptive threshold) |
|---|---|---|
| baseline | 18 found, labels 100%, word diff 1% | 18 found, 0 missed, labels 100%, 1% |
| Student 12 dB quieter | 18 found, 100%, 1% | 18 found, 0 missed, 100%, 1% |
| Student 24 dB quieter | 10 found, 9 missed, 62% | 18 found, 0 missed, 100%, 1% |
| Everyone 30 dB quieter | 0 found | 18 found, 1 missed, 100%, 1% |
| Noise, 20 dB signal-to-noise | crashed | 18 found, 1 missed, 100%, 1% |
| Noise, 10 dB signal-to-noise | crashed | 18 found, 3 missed, 100%, 2% |
| Student 18 dB quieter + noise 20 dB | crashed | 18 found, 1 missed, 100%, 3% |

Changes between the two versions: loudness threshold adapts to the background noise level,
transcription runs in a background worker (the first version lost the connection while
transcribing), a sentence ends at a short pause after a speaker change. The remaining
"missed" turns are probably timing mismatches with a coarse timeline, since the word
difference stays at 1 to 3%.

## 4. Meeting summaries (ROUGE, 5 AMI meetings)

Data: QMSum (Zhong et al., NAACL 2021), Product domain = AMI meetings, whole-meeting
human summaries (100 words on average). Meetings of 2,015 to 8,103 words.
Model: BART-large fine-tuned on SAMSum dialogues (philschmid/bart-large-cnn-samsum).

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |
|---|---|---|---|---|
| First 100 words of the transcript | 13.8 | 2.4 | 8.4 | 100 |
| Single pass (first 1,024 tokens only) | 30.8 | 8.7 | 17.9 | 53 |
| Hierarchical (ours) | 33.8 | 9.3 | 20.2 | 90 |
| Hierarchical, longer final summary | 34.6 | 9.3 | 20.0 | 97 |

Method: 300-token pieces summarized, then summaries of summaries until at most 250 words.
Notes: only 5 meetings, so no significance claim. The model was trained on short chat
dialogues, not meetings. Typical errors: garbled numbers, a hallucinated word, missed topics.

## 5. Named entities (pilot, 28-line synthetic meeting, 34 gold entities)

NER: dslim/bert-base-NER (people, places, organizations, other) plus our own rules for dates
and times. The meeting and the answer key were written by us, so this is a pilot only.

| Text | Match | Precision | Recall | F1 |
|---|---|---|---|---|
| Original | exact type | 0.88 | 0.88 | 0.88 |
| Original | type ignored | 0.97 | 0.97 | 0.97 |
| Lowercase, no punctuation | either | 1.00 | 0.44 | 0.61 |

Per type (exact, original text): dates 13/13, people 7/7, places 2/2 correct; organizations
weakest (3 correct, 2 wrong, 3 missed) because of label disagreements (Razorpay typed as
other, Campus Connect typed as organization) and one span difference (IT / IT office). The
model is case-sensitive, so recall halves on lowercase text; Whisper writes capitals, so the
pipeline is not affected.

## 6. Action items and key dates (rules)

On the same synthetic meeting the rules found all 14 expected action items (who, task, due
date), merged "Sure, I will do it" replies into the assigned task, and listed 9 key dates.
Not an independent test (we wrote both the meeting and the rules). Weakness: tasks that
contain pronouns stay vague ("be ready before that").

## 7. Live delay (end of a sentence to its subtitle)

Setup: the 65 s interview recording played at normal speed to the running server on the
laptop CPU, with speaker changes from the timeline file. 18 sentences. Delay = arrival time
of the subtitle minus the moment the speech of that sentence ended.

| Mean | Median | 90th percentile | Maximum |
|---|---|---|---|
| 2.44 s | 2.23 s | 2.94 s | 4.35 s |

Typical sentence: 2.0 to 2.5 s. A 6 s sentence: 2.75 s. The first sentence took 4.35 s
because the first call loads the speech filter and warms up the model (a warm-up call at
server start would remove this). The delay is made of the pause the server waits for to be
sure the sentence ended (0.4 s after a speaker change, 0.8 s otherwise, so about 0.4 s more
in a call without speaker changes) plus the Whisper `base` time, roughly 0.4 to 0.5 times the
sentence length. Ideas to reduce it: warm-up at start, greedy decoding, interim subtitles
while the sentence is still being spoken.

## 8. Datasets, models and tools

- Whisper (OpenAI) through faster-whisper: tiny, base, small
- BART-large SAMSum summarizer: philschmid/bart-large-cnn-samsum
- NER: dslim/bert-base-NER
- AMI Meeting Corpus (Carletta et al., 2005), Hugging Face version edinburghcstr/ami
- QMSum (Zhong et al., NAACL 2021)
- FastAPI, WebSocket, SQLite, Python 3.13

## 9. Not done yet

- Google Meet integration (active speaker read from the meeting page)
- Tab audio mixed with the microphone, for the speaker's own voice
- Warm-up at server start (first sentence is slow)
- Per-speaker summary, topic segmentation, talk-time statistics