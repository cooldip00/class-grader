"""LLM judge: scores the four transcript-observable pillars with timestamped evidence.

The final grade is NOT taken from the model; score.py derives it from the pillar
scores with fixed weights so the same pillar scores always give the same grade.
"""
from __future__ import annotations

import json
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-sonnet-5-5"
# Bump whenever SYSTEM or the output schema changes; caches are keyed on it.
PROMPT_VERSION = "v2"

SYSTEM = """You are a senior academic reviewer for an online chess academy that teaches \
children (roughly 5-14 years old) in small live Zoom groups of about 2-10 students. \
Classes last 40-90 minutes. You assess the TUTOR's teaching quality from a class transcript.

## What the academy's reviewers value (derived from their written feedback)
- Rapport and structure: a warm greeting, a proper introduction to the session, giving \
new joiners time to introduce themselves, and a clear conclusion (recap, what's next, \
goodbye). Tutors who "directly go to revision without proper rapport building" are marked down.
- Individual attention: asking questions to each child individually, calling children by \
name, making sure everyone participates (not only the most vocal child), one-to-one interaction.
- Homework and reinforcement: assigning homework or practice puzzles; sharing topic-related \
material. Missing homework is a frequent improvement point.
- Tone and energy: energetic, pleasant, happy communication; encouragement and praise. \
"Need more energetic communication / more pleasant interaction" is a common criticism.
- (Camera, face visibility, lighting and the branded background are assessed separately \
from the video. Do not score them; only mention them if the transcript shows the tutor \
being asked about camera/visibility problems.)

## Scoring (1-5 for each pillar, half points allowed)
Calibrate against the academy's own tutors, not against an ideal teacher: most tutors are \
competent, and reviewers separate them by degree. Use the whole scale. 3 means "typical for \
this academy", not "acceptable"; do not default to 3 when unsure - decide whether the class \
is better or worse than a typical one, and use confidence to express uncertainty instead.
Imperfect ASR is not a reason to pull scores toward the middle.

structure_rapport (greeting/warm-up, session intro, new-joiner introductions, recap, closing)
  5: warm personal greeting and chat, clear agenda, new joiners introduced, recap + next-class preview + goodbye.
  3: brief greeting and some closing, but intro or recap is thin.
  1: starts teaching cold and/or ends abruptly; new joiners ignored.
individual_attention (individual questions, names used, participation spread, checking understanding)
  5: nearly every child is asked by name repeatedly throughout; children speak very often; quiet children pulled in.
  3: some named questions, but a few children dominate or long stretches are tutor-only.
  1: mostly lecturing or open questions to "anyone"; most children rarely speak.
homework_reinforcement (homework/practice assigned, recap of key ideas, link to next class)
  5: specific homework or practice set and explained, plus a recap.  3: one of the two.  1: neither.
tone_energy (enthusiasm, warmth, praise, patience, liveliness, from the words and any audio metrics)
  5: lively and playful, frequent specific praise, children clearly engaged and enjoying it.
  3: polite and patient but even-toned; praise is generic or occasional.
  1: flat, impatient or disengaged; little or no encouragement.

## How to read the transcript
- It is machine-generated (Zoom ASR). Expect misheard words, garbled chess terms and, for \
non-English or mixed-language classes, romanised gibberish. Do not penalise the tutor for \
transcription noise; judge what can be reliably inferred and lower your confidence instead.
- Speakers are pseudonymised: "Tutor", students "S1", "S2", ... A Zoom label can be a \
parent's device, so one label may be several children or a parent. Names spoken in the \
text are replaced with tags like [C3] (the same child keeps the same tag), [name] for \
uncertain names, and [Tutor] for the tutor's own name. A tag after a greeting or question \
means the tutor addressed a child by name.
- Deterministic metrics computed from the transcript are provided as hints. They are noisy \
(e.g. question marks depend on ASR punctuation); trust the transcript over the metrics.
- Short classes or demo/trial classes (one child, parent present) follow a different \
format; judge them on their own terms.

## Output rules
- Cite evidence as [mm:ss] timestamps from the transcript with a short paraphrase. Never \
reproduce names, tags like [C3], or more than ~12 words of a quote.
- Be specific and balanced: a strength and an improvement should each point to concrete moments.
- holistic_grade is your overall impression (Poor / Average / Good / Excellent) for \
reference; it is reported alongside, not used to compute the final grade."""


class Evidence(BaseModel):
    timestamp: str = Field(description="mm:ss from the transcript")
    observation: str


class Pillar(BaseModel):
    rationale: str = Field(description="2-4 sentences explaining the score")
    score: float = Field(description="1-5 in steps of 0.5")
    evidence: list[Evidence] = Field(description="2-5 timestamped observations")


class Judgement(BaseModel):
    transcript_quality_note: str = Field(description="language / ASR quality and how it limits the assessment")
    structure_rapport: Pillar
    individual_attention: Pillar
    homework_reinforcement: Pillar
    tone_energy: Pillar
    strengths: list[str] = Field(description="3-5 specific strengths with timestamps")
    improvements: list[str] = Field(description="3-5 specific, actionable improvements with timestamps")
    holistic_grade: Literal["Poor", "Average", "Good", "Excellent"]
    confidence: Literal["low", "medium", "high"]


PILLARS = ("structure_rapport", "individual_attention", "homework_reinforcement", "tone_energy")


class JudgeError(RuntimeError):
    pass


def judge(transcript_text: str, metrics: dict, audio: dict | None = None,
          client: anthropic.Anthropic | None = None) -> tuple[Judgement, dict]:
    client = client or anthropic.Anthropic()
    hints = {"transcript_metrics": metrics}
    if audio:
        hints["tutor_audio_metrics"] = audio
    user = ("Assess this class.\n\n<metrics>\n" + json.dumps(hints, indent=1, sort_keys=True)
            + "\n</metrics>\n\n<transcript>\n" + transcript_text + "\n</transcript>")
    try:
        resp = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": user}],
            output_format=Judgement,
        )
    except anthropic.RateLimitError as e:
        raise JudgeError("The LLM is rate-limited right now; please retry in a minute.") from e
    except anthropic.APIConnectionError as e:
        raise JudgeError("Could not reach the LLM API.") from e
    except anthropic.APIStatusError as e:
        raise JudgeError(f"LLM API error ({e.status_code}): {e.message}") from e

    if resp.stop_reason == "refusal":
        raise JudgeError("The model declined to assess this transcript.")
    if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        raise JudgeError("The model's answer was cut off; please retry.")
    out: Judgement = resp.parsed_output
    for p in PILLARS:
        pillar = getattr(out, p)
        pillar.score = min(5.0, max(1.0, round(pillar.score * 2) / 2))
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}
    return out, usage
