# Plan & progress

**Challenge:** grade a class recording Poor / Average / Good / Excellent and explain why.
**Deadline:** Fri 2026-10-23, 12:00 IST — submit hosted link, public repo, redacted AI
history, ≤300-word explanation via the Google Form.
**Scoring weights:** assessment quality 35% · problem solving & AI use 30% · app & code 25% · clarity 10%.

## Architecture
```
recording (upload / Drive link)
  -> ingest.py   file hash, cache, cleanup
  -> transcript  Zoom .vtt if present, else Groq Whisper + LLM speaker labelling (stt.py)
  -> transcript.py  parse, pseudonymise (Tutor/S1.., names -> [C1]..), deterministic metrics
  -> media.py    LOCAL only: frame type, face, lighting, logo; tutor audio energy
  -> judge.py    Claude Sonnet 5.5 scores 4 pillars (1-5, half points) with [mm:ss] evidence
  -> score.py    fixed weights -> grade; visual cap; confidence
  -> app.py      Gradio: Evaluate | Benchmark | Privacy & method
```
Teaching weights: structure & rapport .25 · individual attention .30 · homework .10 ·
tone & energy .20 (normalised). Thresholds: ≥4.25 Excellent, ≥3.5 Good, ≥2.75 Average.
Visual compliance is a gate, not a weight: ≤1.5 -> Poor (quality issue), ≤3 -> flagged.
Label mapping: weak and quality issue -> Poor. Demo recordings shown but not scored.

## Checklist
- [x] transcript.py — vtt parse, pseudonymise, metrics
- [x] judge.py — structured output, error handling, PROMPT_VERSION
- [x] score.py — deterministic grade
- [x] scripts/benchmark.py — local benchmark, public results without quotes/names
- [x] tests/ — 15 offline tests passing
- [x] v1 and v2 benchmarks archived in `results/`
- [x] media.py — YuNet faces on tutor-speaking frames, camera-off, lighting; audio info only
- [x] benchmark v3/v4 with `--video` (17 tests passing)
- [ ] (stretch) branded-background check — B392 still missed
- [x] stt.py — Groq whisper-large-v3-turbo + Haiku 5.5 tutor/student labels (NOT yet tested live)
- [x] ingest.py — gdown / https link, temp-dir cleanup
- [x] pipeline.py — evaluate(), result cache by file hash + prompt version (tested: B435 avg, 25 s, Average)
- [x] app.py — Gradio 3 tabs, auth from APP_USER/APP_PASS, concurrency 1, deletes uploads
- [x] docs/about.md — How it works & privacy tab
- [x] requirements.txt (gradio pinned via HF README sdk_version 5.38.0, not in requirements)
- [x] local app test in .venv (HTTP 200, 3 tabs, cached B435 report renders; OpenCV 5 YuNet = same B441 numbers)
- [x] test STT path — hosted, weak/B558 without vtt: 185 s, works, but graded Average (3.21)
      vs Poor (2.68) with vtt: attention 3->4, tone 3->3.5 (merged/mislabelled speakers?)
- [x] git init + first commit 9a5df1b (scanned: no keys/quotes/emails in files). GitHub repo created by user (empty).
- [x] pushed to github.com/cooldip00/class-grader (author = GitHub noreply email)
- [x] HF Space CoolDip00/class-grader: ~$5 credits + **CPU upgrade** + 1 h sleep (PRO $9 or
      Streamlit rewrite rejected). Deployed with huggingface_hub upload_folder of `git ls-files`
      (scratchpad deploy.py), NOT git push (HF rejects plain-git binaries like the .onnx).
      Gradio SSR crashed on Spaces (svelte-i18n locale) -> `ssr_mode=False`.
- [x] hosted smoke test: no login 401, wrong pw 400, login 200, 3 tabs, benchmark renders
- [x] hosted e2e average/B435 + vtt: 102 s, **Poor 2.68** (benchmark run: Average 2.79; tone 3->2.5).
      Judge noise at the 2.75 boundary. Decision: NO median-of-3 (3x cost on a small budget,
      Anthropic per-minute limits). The result cache already makes the same upload return the same
      report; score.py now adds a "Borderline" note naming the other grade within ±0.15 of a
      cut-off (18 tests). README: both findings in Limitations, checklist scoring as next step.
- [x] README draft with HF frontmatter (method, results v1-v4, privacy, limitations, run locally)
- [ ] 300-word explanation
- [ ] export + redact AI history (Claude Code + Antigravity); drop screenshots showing keys
- [x] delete scratchpad frames/contact sheets/feature dumps (Fri ~00:30; only scripts remain)
- [ ] submit

## Timeline (IST)
- Thu → 01:30: core pipeline, benchmark, tests, start media.py
- 01:30–07:30 sleep
- 07:30–09:30 media/audio, Groq fallback, Drive input, Gradio, HF deploy
  (if video slips past 09:30: ship with visual "not assessed", weight spread to other pillars)
- 09:30–10:30 end-to-end tests
- 10:30–11:15 README, explanation, AI history export
- 11:15–11:45 submit

## Results
| version | change | exact | within one | LLM holistic exact | cost |
|---|---|---|---|---|---|
| v1 | transcript only, weights fixed before data | 2/10 | 8/10 | 2/10 | $0.45 |
| v2 | calibrated anchors ("3 = typical"), half points | 5/10 | 9/10 | 3/10 | $0.54 |
| v3 | + local video (YuNet faces), visual averaged in at 0.15 | 3/10 | 10/10 | 3/10 | $0 (cached) |
| v4 | visual as a gate: ≤1.5 -> Poor (quality issue), ≤3 flagged only | **6/10** | 9/10 | 3/10 | $0 (cached) |

v3 lesson: most tutors pass the camera checks -> visual 5 lifted every grade. Meeting
the standard isn't good teaching, so v4 makes it a gate (decided after seeing v3; say so).
v4 misses: B392 quality issue (no branded background - not detected) -> Average;
B491 Excellent -> Average; B472 Excellent -> Good; good/B435 -> Average.
Video signals (tutor-speaking frames only): B441 face visible 4% ("face not visible" ✓),
average/B435 face brightness 31/255 ("poor lighting" ✓). Audio loudness did NOT track
labels -> shown in app, not scored or sent to judge.

## Decisions & gotchas
- Pair mp4/vtt by folder + batch ID (`pair_key`); filenames differ.
- `student_turns_per_10min` and homework detection track human labels well.
- Name redaction is best-effort (common Western names can slip through) — say so in README.
- Brotli must be ≥1.2 (httpx decoder error otherwise).
- Local gradio has fastapi/starlette conflicts — use a venv before running app locally.
- No server-side model fallback; API outage -> "retry" message.
- User should top up ~$5 before submission so reviewers don't hit an empty balance.

## Log
- Thu 22:20 v1 benchmark. 22:24 v2 benchmark + tests. Next: media.py.
- Thu ~23:40 media.py (Haar failed -> YuNet), v3 averaged 3/10, v4 gate 6/10. Next: ingest, stt, app.
- Fri ~00:30 ingest/stt/pipeline/app done, .venv (gradio 5.38.0), 17 tests pass, app runs locally.
  app.py only deletes uploads inside the Gradio temp dir. Waiting on user: rotate keys, GitHub
  repo, HF Space + secrets, HF_TOKEN in .env. Next (07:30): STT clip test, HF README frontmatter,
  git push, deploy, hosted e2e, README, explanation, history redaction.
- Thu ~23:55 pushed to GitHub, deployed Space (SSR fix), hosted smoke + e2e (vtt and STT) pass.
  Found: judge variance flips boundary grades; STT path over-scores attention. Next: decide on
  judge variance fix, 300-word explanation, history redaction, scratchpad cleanup.
- Fri ~00:30 borderline note deployed, scratchpad images deleted. Anthropic credit $3.63 (~70
  gradings). Next (07:30): explanation, history export + redaction (user exports Antigravity
  chat), final check, submit by 11:00 with APP_USER/APP_PASS in the form.
