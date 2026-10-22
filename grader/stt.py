"""Speech-to-text fallback for recordings that come without a Zoom .vtt transcript.

Audio is cut into 20-minute mono 32 kbps MP3 chunks (~5 MB each, under Groq's upload
limit) and transcribed with Whisper large-v3-turbo on Groq. Whisper gives no speaker
labels, so a small, cheap model labels each line as tutor or student. That is less
reliable than Zoom's labels and every student becomes one speaker, so per-student
metrics are unavailable and confidence is lowered.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from .media import _ffmpeg
from .transcript import Cue

CHUNK_SEC = 20 * 60
WHISPER = "whisper-large-v3-turbo"
LABEL_MODEL = "claude-haiku-5-5"


class STTError(RuntimeError):
    pass


def duration_sec(path: str) -> float:
    err = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
    if not m:
        raise STTError("Could not read the recording's duration; is it a valid video/audio file?")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _chunk(path: str, start: float, out: Path) -> None:
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(CHUNK_SEC), "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                    "-b:a", "32k", str(out)], check=True)


def transcribe(path: str, progress=None) -> tuple[list[dict], str | None]:
    """Return Whisper segments [{start, end, text}] and the detected language."""
    from groq import Groq, GroqError

    client = Groq(max_retries=4)
    total = duration_sec(path)
    segments, language = [], None
    with tempfile.TemporaryDirectory() as tmp:
        starts = range(0, int(total) + 1, CHUNK_SEC)
        for i, start in enumerate(starts):
            if progress:
                progress(i / len(starts), f"Transcribing audio {i + 1}/{len(starts)}")
            mp3 = Path(tmp) / f"chunk{i}.mp3"
            _chunk(path, start, mp3)
            try:
                resp = client.audio.transcriptions.create(
                    model=WHISPER, file=(mp3.name, mp3.read_bytes()), response_format="verbose_json",
                    timestamp_granularities=["segment"], temperature=0.0)
            except GroqError as e:
                raise STTError(f"Speech-to-text failed: {e}") from e
            data = resp.model_dump()
            language = language or data.get("language")
            for s in data.get("segments") or []:
                if (s.get("no_speech_prob") or 0) > 0.8 or not s.get("text", "").strip():
                    continue
                segments.append({"start": start + s["start"], "end": start + s["end"],
                                 "text": s["text"].strip()})
    return segments, language


class Labels(BaseModel):
    labels: str = Field(description="exactly one character per numbered line: T = tutor, S = student/parent")


LABEL_SYSTEM = """You label lines of an automatic transcript of an online chess class for \
children. One tutor teaches a small group of children; parents occasionally speak. For each \
numbered line decide who most likely said it: T for the tutor (explains, asks questions, \
gives instructions, praises, manages the class) or S for a child or parent (short answers, \
moves, questions to the tutor). Consecutive lines are often the same speaker. Return a string \
with exactly one character per line, in order, using only T and S."""


def label_speakers(segments: list[dict], client: anthropic.Anthropic | None = None) -> list[Cue]:
    client = client or anthropic.Anthropic()
    lines = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(segments))
    resp = client.messages.parse(
        model=LABEL_MODEL, max_tokens=8000,
        system=LABEL_SYSTEM,
        messages=[{"role": "user", "content": f"{len(segments)} lines:\n{lines}"}],
        output_format=Labels,
    )
    raw = re.sub(r"[^TS]", "", (resp.parsed_output.labels if resp.parsed_output else "").upper())
    # Pad or trim rather than fail: a slightly misaligned label string is still mostly right.
    raw = (raw + raw[-1:] * len(segments))[: len(segments)] if raw else "T" * len(segments)
    return [Cue(s["start"], s["end"], "Tutor" if lab == "T" else "S1", s["text"])
            for s, lab in zip(segments, raw)]
