---
title: Class Grader
emoji: ♟️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.38.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Grades a recorded online class and explains why
---

# Class Grader

Upload a recorded online chess class (or paste a Google Drive link) and get a grade of
**Poor / Average / Good / Excellent**, with pillar scores and timestamped evidence explaining it.

- **Live app:** https://huggingface.co/spaces/CoolDip00/class-grader (login required; credentials are in the submission form)
- **Benchmark:** 6/10 exact and 9/10 within one grade on the 10 labelled sample recordings

## How it works

```
recording (upload or Drive / https link)
  -> transcript   Zoom .vtt if provided, else Groq Whisper + Claude Haiku tutor/student labels
  -> pseudonymise speakers -> Tutor, S1, S2 ...; children's names in text -> [C1] ...
  -> metrics      talk share, student turns, questions, praise, names used, greeting/recap/homework
  -> camera check LOCAL: 1 frame / 10 s, OpenCV YuNet face detector on frames where the tutor speaks
  -> judge        Claude Sonnet 5.5 scores 4 teaching pillars (1-5) with [mm:ss] evidence
  -> grade        computed in code from fixed weights; camera problems act as a gate
```

The rubric comes from the academy's own reviewer feedback on the sample classes:

| Pillar | High score means | Weight |
|---|---|---|
| Structure & rapport | Greeting, intro, new joiners welcomed, recap, goodbye | 25% |
| Individual attention | Children asked by name, everyone participates, understanding checked | 30% |
| Homework & reinforcement | Specific practice set, key ideas recapped | 10% |
| Tone & energy | Lively, encouraging, specific praise, patient | 20% |
| Camera compliance | Face visible, well lit, camera on | gate |

Weighted score ≥ 4.25 Excellent, ≥ 3.5 Good, ≥ 2.75 Average, else Poor. A serious camera
failure (score ≤ 1.5) makes the class a *quality issue* and grades it Poor whatever the
teaching; smaller camera problems are reported but not graded. The LLM never picks the grade
itself, so the same pillar scores always give the same grade.

Confidence is lowered for mostly non-English transcripts, when no Zoom transcript is
available, and near a grade boundary.

## Results

Evaluated on the 10 labelled sample recordings (weak and quality issue both count as Poor).
Each version is archived in [`results/`](results/).

| Version | Change | Exact | Within one grade |
|---|---|---|---|
| v1 | Transcript only; weights fixed before seeing any results | 2/10 | 8/10 |
| v2 | Judge calibrated to the academy ("3 = typical tutor"), half points | 5/10 | 9/10 |
| v3 | + local video checks averaged in at 15% | 3/10 | 10/10 |
| v4 | Camera as a quality gate instead of a pillar | **6/10** | 9/10 |

What I learned along the way:

- **v1 → v2:** the uncalibrated judge put almost every tutor at 3-4. Anchoring "3" to a typical
  tutor at this academy and asking for the whole scale spread the scores out.
- **v2 → v3:** most tutors pass the camera checks, so averaging a 5/5 camera score in lifted
  every grade. Meeting the camera standard is expected, not a sign of good teaching.
- **v3 → v4:** treating the camera as a gate matches how the academy labels "quality issue"
  classes. This change was made *after* seeing v3's results on the same 10 recordings, so 6/10
  is an optimistic estimate.
- The video signals are real: in the quality-issue class B441 the tutor's face was visible in
  4% of their speaking time (reviewer: "face not visible"); in average/B435 the face brightness
  was 31/255 (reviewer: "poor lighting").
- Audio loudness did not track the labels, so it is shown in the report but not scored.
- Remaining misses: one quality-issue class whose problem was a missing branded background
  (not detected), and two Excellent classes graded one step lower.

## Privacy

- **Video frames never leave the server.** Face and lighting checks run locally.
- **Anthropic receives only the pseudonymised transcript.** **Groq receives audio** only when
  no `.vtt` is provided, for speech-to-text.
- Name removal is best-effort: a common name misheard by speech-to-text can slip through.
- Uploads and downloads are deleted after grading. Reports are cached by file hash so a
  recording is not processed and billed twice.
- The app is behind a login. The confidential recordings are not in this repository; the
  published benchmark contains only batch IDs, labels and scores, with no quotes or evidence.

## Limitations

- Tuned and tested on 10 labelled recordings: a sanity check, not a validation.
- A missing branded background is not detected.
- The judge varies by about ±0.15 points between runs on the same class.
- Without a `.vtt`, all students are merged into one speaker, so participation spread is approximate.
- Non-English or mixed-language classes get low confidence.
- The hosted Space sleeps when idle; the first request after a sleep takes about a minute.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt gradio==5.38.0 pytest
# .env (in this folder or a parent): ANTHROPIC_API_KEY, GROQ_API_KEY, optional APP_USER / APP_PASS
.\.venv\Scripts\python -m pytest tests -q
.\.venv\Scripts\python app.py
```

Benchmark (needs the sample recordings, which are not in the repo):

```powershell
python scripts\benchmark.py "<folder with average/ excellent/ good/ ... subfolders>" --workers 3 --video
```

## Repository layout

| Path | What it does |
|---|---|
| `app.py` | Gradio app: Evaluate, Benchmark, How it works & privacy |
| `grader/transcript.py` | vtt parsing, pseudonymisation, deterministic metrics |
| `grader/stt.py` | Groq Whisper transcription + Claude Haiku speaker labels |
| `grader/media.py` | Local frame sampling, YuNet face / camera-off / lighting checks |
| `grader/judge.py` | Claude Sonnet 5.5 structured judgement (`PROMPT_VERSION`) |
| `grader/score.py` | Weights, thresholds, camera gate, confidence |
| `grader/pipeline.py` | `evaluate()`: ties it together, with a result cache |
| `grader/ingest.py` | Drive / https download, temp-file cleanup |
| `scripts/benchmark.py` | Runs the labelled benchmark and writes `benchmark_results.json` |
| `PLAN.md`, `CLAUDE.md` | The working plan and rules used while building with an AI coding agent |

Face detection uses [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
(`face_detection_yunet_2023mar.onnx`, MIT licence).
