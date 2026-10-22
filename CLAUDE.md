# class-grader — working rules

Read `PLAN.md` first: it is the source of truth for status, decisions and next steps.
Update its checklist and log whenever a task is finished or a decision changes.

## Hard rules
- Recordings are confidential. Never commit or upload mp4/vtt/pdf/frames; never send video
  frames to any API. Only pseudonymised transcript text goes to Anthropic / Groq.
- No tutor or child names anywhere in the repo, app, results or docs. Use batch IDs (B482).
- Never read or print `.env` (lives in the parent folder, loaded via python-dotenv).
  Never hard-code keys or the app password; the app reads them from env / HF Space secrets.
- Never add a Co-Authored-By trailer to commits.
- Final grade is computed in code (`grader/score.py`), not taken from the LLM.
- Bump `PROMPT_VERSION` in `grader/judge.py` whenever the prompt or schema changes.
- API budget is small (~$3.75 prepaid): don't `--rejudge` the benchmark without reason.

## Commands
- Tests: `python -m pytest tests -q`
- Benchmark: `python scripts\benchmark.py "d:\Greylabs AI\class recordings -20261022T150352Z-1-001\class recordings" --workers 3 [--video] [--only B482]`
