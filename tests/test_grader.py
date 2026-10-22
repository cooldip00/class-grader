"""Offline tests: no API calls, no real recordings. All names below are invented."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import score as S  # noqa: E402
from grader import transcript as T  # noqa: E402
from scripts.benchmark import pair_key  # noqa: E402

VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
Coach Ravindran: Good evening everyone! Hi Zorvika, how are you?

2
00:00:05.000 --> 00:00:06.500
Zorvika's iPad: I am fine.

3
00:00:07.000 --> 00:00:12.000
Coach Ravindran: Today we will learn the fork. Tavrin, what is a fork?

4
00:00:13.000 --> 00:00:15.000
Tavrin Mehl: When one piece attacks two pieces.

5
00:00:16.000 --> 00:00:40.000
Coach Ravindran: Very good! Excellent answer. For homework, please solve five fork puzzles.
Bye everyone, see you next class.
"""


@pytest.fixture
def tr():
    return T.load(VTT)


def test_parse_vtt_speakers_and_multiline_cue():
    cues = T.parse_vtt(VTT)
    assert len(cues) == 5
    assert cues[0].speaker == "Coach Ravindran"
    assert cues[0].start == 1.0 and cues[4].end == 40.0
    assert "Bye everyone" in cues[4].text  # continuation line joined into the same cue


def test_parse_vtt_without_speaker_label():
    cues = T.parse_vtt("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nno label here\n")
    assert cues[0].speaker == "Unknown" and cues[0].text == "no label here"


def test_pseudonymise_labels_and_spoken_names(tr):
    assert [c.speaker for c in tr.cues] == ["Tutor", "S1", "Tutor", "S2", "Tutor"]
    text = T.render(tr)
    for real in ("Ravindran", "Zorvika", "Tavrin", "Mehl"):
        assert real not in text
    assert "[C" in text  # children replaced by stable tags


def test_same_child_keeps_same_tag(tr):
    greeted = tr.cues[0].text
    tag = greeted[greeted.index("[C"):greeted.index("]", greeted.index("[C")) + 1]
    assert tag.startswith("[C")
    assert T.render(tr).count(tag) >= 1


def test_metrics_detect_structure_and_homework(tr):
    m = T.metrics(tr)
    assert m["greeting_in_first_8min"]
    assert m["closing_in_last_8min"]
    assert m["homework_mentions"]
    assert m["student_speakers_active"] == 0  # nobody has 3+ turns in this tiny class
    assert 0 < m["tutor_talk_share"] < 1


def test_empty_transcript():
    assert T.load("WEBVTT\n").cues == []


@pytest.mark.parametrize("name,key", [
    ("B482 - Chess class 2026-10-01.mp4", "b482"),
    ("B482-Chess class.vtt", "b482"),
    ("demo english.mp4", "demo english"),
    ("Demo English - transcript.vtt", "demo english"),
])
def test_pair_key_matches_mp4_and_vtt(name, key):
    assert pair_key(name) == key


def test_grade_thresholds():
    all_ = lambda v: {k: v for k in S.WEIGHTS}  # noqa: E731
    assert S.grade(all_(5))["grade"] == "Excellent"
    assert S.grade(all_(4))["grade"] == "Good"
    assert S.grade(all_(3))["grade"] == "Average"
    assert S.grade(all_(2))["grade"] == "Poor"


def test_missing_pillar_renormalises_weights():
    p = {k: 4.0 for k in S.WEIGHTS}
    p["homework_reinforcement"] = None
    r = S.grade(p)
    assert r["weighted_score"] == 4.0
    assert "homework_reinforcement" not in r["weights_used"]
    assert abs(sum(r["weights_used"].values()) - 1) < 1e-3
    assert any("not assessed" in n for n in r["notes"])  # no visual given


def test_visual_failure_is_quality_issue():
    r = S.grade({k: 5.0 for k in S.WEIGHTS}, visual=1.5)
    assert r["grade"] == "Poor" and r["teaching_grade"] == "Excellent"
    assert any("Quality issue" in n for n in r["notes"])


def test_visual_concern_does_not_change_grade():
    r = S.grade({k: 4.0 for k in S.WEIGHTS}, visual=3.0)
    assert r["grade"] == "Good"
    assert any("concern" in n for n in r["notes"])
    assert S.grade({k: 4.0 for k in S.WEIGHTS}, visual=5.0)["notes"] == []


def test_visual_score_deductions():
    from grader import media as M
    ok = {"tutor_speaking_frames": 50, "tutor_face_visible_share": 0.9,
          "tutor_camera_off_share": 0.0, "tutor_face_brightness": 110}
    assert M.visual_score(ok) == (5.0, [])
    hidden, why = M.visual_score({**ok, "tutor_face_visible_share": 0.04})
    assert hidden == 1.5 and "4%" in why[0]
    dark, _ = M.visual_score({**ok, "tutor_face_brightness": 31})
    assert dark == 3.0
    assert M.visual_score({**ok, "tutor_speaking_frames": 2})[0] is None


def test_borderline_names_the_other_grade():
    # hosted B435 run: 2.68 just below the Average cut-off
    p = {"structure_rapport": 2.5, "individual_attention": 3.0, "homework_reinforcement": 2.5, "tone_energy": 2.5}
    r = S.grade(p, llm_confidence="high", visual=5.0)
    assert r["grade"] == "Poor" and r["confidence"] == "medium"
    assert any("Borderline" in n and "could also be graded Average" in n for n in r["notes"])
    above = S.grade({k: 3.6 for k in S.WEIGHTS}, visual=5.0)  # just above Good
    assert any("could also be graded Average" in n for n in above["notes"])
    # quality-issue override: no borderline note, the grade is fixed by the camera gate
    assert not any("Borderline" in n for n in S.grade(p, visual=1.0)["notes"])


def test_low_english_lowers_confidence():
    r = S.grade({k: 4.0 for k in S.WEIGHTS}, english_ratio=0.3, llm_confidence="high")
    assert r["confidence"] == "low"


def test_labels_map_to_four_grades():
    assert set(S.LABELS.values()) == set(S.GRADES)
