"""End-to-end evaluation of one recording: transcript -> metrics -> video -> judge -> grade."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import judge as J
from . import media as M
from . import score as S
from . import stt
from . import transcript as T
from .ingest import file_hash

CACHE = Path(os.getenv("GRADER_CACHE", Path(__file__).resolve().parents[1] / ".cache" / "results"))


class PipelineError(RuntimeError):
    pass


def _noop(_frac, _msg):
    pass


def evaluate(video: str, vtt_text: str | None = None, progress=_noop, client=None) -> dict:
    """Grade one recording. Results are cached by file content + prompt version, so
    re-submitting the same recording is instant and costs nothing."""
    progress(0.02, "Hashing file")
    key = file_hash(video)
    if vtt_text:
        key += hashlib.sha256(vtt_text.encode()).hexdigest()[:16]
    cache_file = CACHE / J.PROMPT_VERSION / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text(encoding="utf-8")), "cached": True}

    if vtt_text:
        progress(0.05, "Reading Zoom transcript")
        tr, source, language = T.load(vtt_text), "zoom_vtt", None
    else:
        segs, language = stt.transcribe(video, lambda f, m: progress(0.05 + 0.25 * f, m))
        if not segs:
            raise PipelineError("No speech was found in the recording.")
        progress(0.32, "Labelling tutor and student lines")
        tr, source = T.pseudonymise(stt.label_speakers(segs, client)), "speech_to_text"
    if not tr.cues:
        raise PipelineError("The transcript is empty.")

    m = T.metrics(tr)
    if source == "speech_to_text":
        m["transcript_source"] = ("automatic speech-to-text; tutor/student labels were inferred, "
                                  "all students are merged into S1")

    progress(0.4, "Checking the video locally (camera, face, lighting)")
    media = M.analyse(video, tr)

    progress(0.6, "Assessing teaching quality (about 20-40 s)")
    jd, usage = J.judge(T.render(tr), m, client=client)

    pillars = {k: getattr(jd, k).score for k in J.PILLARS}
    result = S.grade(pillars, m.get("english_ratio"), jd.confidence, media["visual_score"])
    if source == "speech_to_text" and result["confidence"] == "high":
        result["confidence"] = "medium"
    if source == "speech_to_text":
        result["notes"].append("No Zoom transcript was provided: speech-to-text with inferred "
                               "speaker labels was used, so per-student participation is approximate.")

    out = {
        **result,
        "pillars": pillars,
        "visual_score": media["visual_score"],
        "visual_reasons": media["visual_reasons"],
        "visual": {k: v for k, v in media.items() if k not in ("audio", "timeline", "visual_score", "visual_reasons")},
        "audio": media["audio"],
        "timeline": [{k: r[k] for k in ("t", "speaker", "face", "camera_off")} for r in media["timeline"]],
        "judgement": jd.model_dump(),
        "metrics": m,
        "source": source,
        "language": language,
        "usage": usage,
        "prompt_version": J.PROMPT_VERSION,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out, indent=1), encoding="utf-8")
    progress(1.0, "Done")
    return {**out, "cached": False}
