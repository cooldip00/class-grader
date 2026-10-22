"""Local video and audio checks. Nothing here calls an API: frames and audio never
leave the machine, only the numbers computed from them do.

Zoom records in active-speaker view: a frame shows whoever is talking, either full
screen or, while a board is screen-shared, as a thumbnail in the top-right corner.
So each sampled frame is matched to the transcript speaker at that moment, and the
camera checks use only frames where the tutor is speaking. Whole-video averages would
mostly measure how long the screen was shared or how the children's cameras looked.

Faces are found with OpenCV's YuNet model (bundled, ~230 KB, runs on CPU); it also
finds the small speaker thumbnail during screen share, which Haar cascades miss.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .transcript import Transcript

FRAME_EVERY = 10          # seconds between sampled frames
W, H = 960, 540           # analysis resolution (thumbnail faces are ~30 px wide here)
AUDIO_SR = 16000
WIN = AUDIO_SR // 2       # 0.5 s loudness windows
YUNET = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"

# Thresholds were set by looking at the per-class numbers of the labelled recordings
# (see README); with 10 recordings they are a starting point, not a fitted model.
FACE_SHARE_CUTS = ((0.1, 3.5), (0.3, 2.0), (0.6, 1.0))     # (below, penalty)
DARK_FACE_CUTS = ((45, 2.0), (60, 1.0))                    # median face brightness 0-255
CAMERA_OFF_CUTS = ((0.3, 2.0), (0.12, 1.0))                # (above, penalty)
MIN_TUTOR_FRAMES = 6


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def iter_frames(path: str, every: int = FRAME_EVERY):
    """Yield (seconds, BGR frame) every `every` seconds, decoding keyframes only."""
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-i", path,
           "-vf", f"fps=1/{every},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = W * H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        i = 0
        while len(buf := proc.stdout.read(size)) == size:
            yield i * every, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def frame_features(frame: np.ndarray, detector) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, faces = detector.detect(frame)
    out = {"camera_off": bool((gray < 45).mean() > 0.9), "face": None, "face_brightness": None}
    if faces is not None and len(faces):
        x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
        out["face"] = "full" if w / W >= 0.12 else "thumbnail"
        out["face_brightness"] = float(gray[max(y, 0):y + h, max(x, 0):x + w].mean())
    return out


def speaker_at(tr: Transcript, t: float, slack: float = 2.0) -> str | None:
    """Transcript speaker at time t (the active-speaker tile lags speech slightly)."""
    best = None
    for c in tr.cues:
        if c.start - 0.5 <= t <= c.end + slack:
            best = c.speaker
        elif c.start > t:
            break
    return best


def audio_energy(path: str, tr: Transcript) -> dict | None:
    """Loudness statistics of the tutor's speech, from 0.5 s windows inside tutor cues.
    Reported for information only: on the labelled set they did not track the grades."""
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(AUDIO_SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False).stdout
    if not raw:
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    n = len(x) // WIN
    db = 20 * np.log10(np.sqrt((x[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1)) + 1e-6)
    tutor = np.zeros(n, bool)
    words = 0
    for c in tr.cues:
        if c.speaker == "Tutor":
            tutor[int(c.start * 2): int(c.end * 2) + 1] = True
            words += len(c.text.split())
    voiced = tutor & (db > -50)
    if voiced.sum() < 60:
        return None
    v = db[voiced]
    return {
        "tutor_loudness_db": round(float(np.median(v)), 1),
        "tutor_dynamic_range_db": round(float(np.percentile(v, 90) - np.percentile(v, 10)), 1),
        "tutor_words_per_min": round(float(words / (tutor[:n].sum() / 120)), 1),
        "silent_share_of_class": round(float((db < -50).mean()), 2),
    }


def visual_summary(rows: list[dict]) -> dict:
    tutor = [r for r in rows if r["speaker"] == "Tutor"]
    off = [r for r in tutor if r["camera_off"]]
    faces = [r for r in tutor if r["face"]]
    lit = [r["face_brightness"] for r in faces]
    n = len(tutor)
    return {
        "frames_sampled": len(rows),
        "tutor_speaking_frames": n,
        "tutor_face_visible_share": round(len(faces) / n, 2) if n else None,
        "tutor_camera_off_share": round(len(off) / n, 2) if n else None,
        "tutor_face_brightness": round(float(np.median(lit)), 1) if lit else None,
        "tutor_on_full_camera_share": round(sum(r["face"] == "full" for r in tutor) / n, 2) if n else None,
    }


def visual_score(s: dict) -> tuple[float | None, list[str]]:
    """1-5 visual compliance plus the reasons for each deduction."""
    if s["tutor_speaking_frames"] < MIN_TUTOR_FRAMES:
        return None, ["Too little footage of the tutor speaking to assess the camera."]
    score, why = 5.0, []
    fv = s["tutor_face_visible_share"]
    for cut, pen in FACE_SHARE_CUTS:
        if fv < cut:
            score -= pen
            why.append(f"Tutor's face visible in only {fv:.0%} of the moments they were speaking.")
            break
    off = s["tutor_camera_off_share"]
    for cut, pen in CAMERA_OFF_CUTS:
        if off > cut:
            score -= pen
            why.append(f"Camera off or screen black in {off:.0%} of the tutor's speaking moments.")
            break
    fb = s["tutor_face_brightness"]
    if fb is not None:
        for cut, pen in DARK_FACE_CUTS:
            if fb < cut:
                score -= pen
                why.append(f"Face is poorly lit (median brightness {fb:.0f}/255).")
                break
    return max(1.0, score), why


def analyse(path: str, tr: Transcript, every: int = FRAME_EVERY) -> dict:
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (W, H), score_threshold=0.7)
    rows = []
    for t, frame in iter_frames(path, every):
        rows.append({"t": t, "speaker": speaker_at(tr, t), **frame_features(frame, detector)})
    summary = visual_summary(rows)
    score, reasons = visual_score(summary)
    return {
        "visual_score": score,
        "visual_reasons": reasons,
        **summary,
        "audio": audio_energy(path, tr),
        "timeline": rows,
    }
