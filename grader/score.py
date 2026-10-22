"""Deterministic grade from pillar scores.

The teaching weights and thresholds were fixed before running the benchmark, so they
are not fitted to the 10 labelled recordings. Visual compliance is not averaged in:
meeting the camera standard is expected, not a sign of good teaching, and averaging a
"5 - no problems found" lifted every grade (benchmark v3). It works as a gate instead,
mirroring the academy's separate "quality issue" category. That change was made after
seeing the v3 results and is reported as such.
"""
from __future__ import annotations

GRADES = ("Poor", "Average", "Good", "Excellent")

WEIGHTS = {
    "structure_rapport": 0.25,
    "individual_attention": 0.30,
    "homework_reinforcement": 0.10,
    "tone_energy": 0.20,
}
# Weighted mean on the 1-5 scale -> grade.
THRESHOLDS = ((4.25, "Excellent"), (3.5, "Good"), (2.75, "Average"))
# Visual compliance (1-5 from media.py): at or below FAIL the class is a quality issue
# whatever the teaching; at or below CONCERN the problem is reported but not graded.
VISUAL_FAIL = 1.5
VISUAL_CONCERN = 3.0

LABELS = {"excellent": "Excellent", "good": "Good", "average": "Average",
          "weak": "Poor", "quality issue": "Poor"}


def grade(pillars: dict[str, float | None], english_ratio: float | None = None,
          llm_confidence: str = "medium", visual: float | None = None) -> dict:
    used = {k: v for k, v in pillars.items() if k in WEIGHTS and v is not None}
    total_w = sum(WEIGHTS[k] for k in used)
    weighted = sum(WEIGHTS[k] * v for k, v in used.items()) / total_w
    teaching = next((name for cut, name in THRESHOLDS if weighted >= cut), "Poor")
    g = teaching

    notes = []
    if visual is None:
        notes.append("Camera compliance not assessed (no video or too little footage of the tutor).")
    elif visual <= VISUAL_FAIL:
        g = "Poor"
        notes.append(f"Quality issue: serious camera problems (visual score {visual}/5), so the class "
                     f"is graded Poor regardless of teaching (teaching alone: {teaching}).")
    elif visual <= VISUAL_CONCERN:
        notes.append(f"Camera concern (visual score {visual}/5): reported, not counted in the grade.")

    conf = llm_confidence
    if english_ratio is not None and english_ratio < 0.55:
        conf = "low"
        notes.append("Transcript is largely non-English or garbled ASR; teaching quality is hard to judge from text.")
    # Close to a boundary -> say so rather than overstate certainty.
    if g == teaching and any(abs(weighted - cut) < 0.1 for cut, _ in THRESHOLDS) and conf == "high":
        conf = "medium"
    return {"grade": g, "teaching_grade": teaching, "weighted_score": round(weighted, 2),
            "confidence": conf, "notes": notes,
            "weights_used": {k: round(WEIGHTS[k] / total_w, 3) for k in used}}
