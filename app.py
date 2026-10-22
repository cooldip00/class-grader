"""Gradio app: upload a class recording (or paste a link) and get a graded, explained report."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from grader import ingest  # noqa: E402
from grader.judge import JudgeError  # noqa: E402
from grader.pipeline import PipelineError, evaluate  # noqa: E402
from grader.stt import STTError  # noqa: E402

ROOT = Path(__file__).parent
TEMP = Path(os.getenv("GRADIO_TEMP_DIR") or tempfile.gettempdir()).resolve()
PILLAR_NAMES = {
    "structure_rapport": "Structure & rapport",
    "individual_attention": "Individual attention",
    "homework_reinforcement": "Homework & reinforcement",
    "tone_energy": "Tone & energy",
}
BADGE = {"Excellent": "🟢", "Good": "🔵", "Average": "🟠", "Poor": "🔴"}
VERSION_NOTES = {
    "v1": "Transcript only; weights fixed before seeing any results",
    "v2": "Judge calibrated to the academy ('3 = typical tutor'), half points",
    "v3": "+ local video checks, averaged in as a 15% pillar (lifted every grade)",
    "v4": "Camera as a quality gate: serious failure = Poor, smaller issues flagged",
}


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def render_report(r: dict) -> tuple[str, str, str]:
    jd = r["judgement"]
    head = [f"# {BADGE[r['grade']]} {r['grade']}",
            f"**Confidence:** {r['confidence']} · **Teaching score:** {r['weighted_score']} / 5"
            + (f" (teaching alone: {r['teaching_grade']})" if r["teaching_grade"] != r["grade"] else ""),
            ""]
    head += [f"> {n}" for n in r["notes"]]
    head += ["", "| Pillar | Score | Weight |", "|---|---|---|"]
    for k, name in PILLAR_NAMES.items():
        head.append(f"| {name} | {r['pillars'][k]} / 5 | {r['weights_used'].get(k, 0):.0%} |")
    vis = r["visual_score"]
    head.append(f"| Camera compliance (gate) | {vis if vis is not None else 'not assessed'}"
                f"{' / 5' if vis is not None else ''} | pass / flag / fail |")
    head += ["", "### Strengths", *[f"- {s}" for s in jd["strengths"]],
             "", "### What to improve", *[f"- {s}" for s in jd["improvements"]]]

    ev = []
    for k, name in PILLAR_NAMES.items():
        p = jd[k]
        ev += [f"### {name} — {p['score']} / 5", p["rationale"], ""]
        ev += [f"- **[{e['timestamp']}]** {e['observation']}" for e in p["evidence"]]
        ev.append("")
    ev += ["### Camera & video (checked locally, frames never leave the server)"]
    ev += [f"- {w}" for w in r["visual_reasons"]] or ["- No camera problems found."]
    v = r["visual"]
    if v.get("tutor_speaking_frames"):
        ev.append(f"- Sampled {v['frames_sampled']} frames; in {v['tutor_speaking_frames']} the tutor was "
                  f"speaking. Face visible: {v['tutor_face_visible_share']:.0%}; camera off: "
                  f"{v['tutor_camera_off_share']:.0%}; face brightness: {v['tutor_face_brightness']}/255.")
    if r.get("audio"):
        a = r["audio"]
        ev.append(f"- Audio (informational, not scored): tutor speaks ~{a['tutor_words_per_min']} words/min, "
                  f"loudness range {a['tutor_dynamic_range_db']} dB.")
    ev += ["", f"*Transcript note:* {jd['transcript_quality_note']}"]

    m = r["metrics"]
    keys = ["duration_min", "tutor_talk_share", "student_speakers_active", "student_turns_per_10min",
            "tutor_questions_per_10min", "distinct_children_addressed_by_name", "praise_per_10min",
            "longest_tutor_monologue_min", "greeting_in_first_8min", "recap_in_last_8min",
            "homework_mentions", "english_ratio"]
    met = ["| Metric | Value |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {m.get(k)} |" for k in keys if k in m]
    met += ["", f"Source: {'Zoom transcript' if r['source'] == 'zoom_vtt' else 'speech-to-text'} · "
            f"prompt {r['prompt_version']}" + (" · served from cache" if r.get("cached") else "")]
    return "\n".join(head), "\n".join(ev), "\n".join(met)


def run(upload, link, vtt, progress=gr.Progress()):
    if not upload and not (link or "").strip():
        raise gr.Error("Upload a recording or paste a link.")
    downloaded = None
    try:
        if upload:
            path = upload
        else:
            progress(0.01, desc="Downloading recording")
            path = downloaded = ingest.fetch(link)
        vtt_text = Path(vtt).read_text(encoding="utf-8", errors="replace") if vtt else None
        r = evaluate(path, vtt_text, lambda f, msg: progress(f, desc=msg))
        return render_report(r)
    except (ingest.IngestError, PipelineError, JudgeError, STTError) as e:
        raise gr.Error(str(e)) from e
    finally:
        ingest.cleanup(downloaded)
        for p in (upload, vtt):  # don't keep confidential uploads around (only Gradio's temp copies)
            if p and os.path.exists(p) and Path(p).resolve().is_relative_to(TEMP):
                os.remove(p)


def benchmark_md() -> str:
    f = ROOT / "benchmark_results.json"
    if not f.exists():
        return "Benchmark results not found."
    data = json.loads(f.read_text(encoding="utf-8"))
    s = data["summary"]
    rows = ["| Recording | Human label | Predicted | Teaching score | Camera | Judge's own grade |",
            "|---|---|---|---|---|---|"]
    for r in data["rows"]:
        rows.append(f"| {r['id']} | {r['label'] or '(demo, unlabelled)'} | {r['predicted']} | "
                    f"{r['weighted_score']} | {r['pillars'].get('visual_compliance') or '-'} | {r['llm_holistic']} |")
    hist = ["| Version | Change | Exact | Within one grade |", "|---|---|---|---|"]
    for p in sorted((ROOT / "results").glob("v*.json")):
        sm = json.loads(p.read_text(encoding="utf-8"))["summary"]
        hist.append(f"| {p.stem.split('_')[0]} | {VERSION_NOTES.get(p.stem.split('_')[0], '')} | "
                    f"{sm['exact']}/{sm['n_labelled']} | {sm['within_one']}/{sm['n_labelled']} |")
    return (f"**Current:** {s['exact']}/{s['n_labelled']} exact, {s['within_one']}/{s['n_labelled']} within one "
            f"grade on the 10 labelled recordings (weak and quality issue = Poor).\n\n"
            + "\n".join(rows) + "\n\n### How it got here\n" + "\n".join(hist)
            + "\n\nNo quotes, names or evidence from the confidential recordings are published; "
              "only batch IDs, labels and scores.")


ABOUT = (ROOT / "docs" / "about.md").read_text(encoding="utf-8") if (ROOT / "docs" / "about.md").exists() else ""

with gr.Blocks(title="Class Grader", theme=gr.themes.Soft(), delete_cache=(1800, 1800)) as demo:
    gr.Markdown("# Class Grader\nGrades a recorded online chess class as **Poor / Average / Good / "
                "Excellent** and explains why, with timestamped evidence.")
    with gr.Tab("Evaluate"):
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Class recording (.mp4)", file_types=["video", "audio"], type="filepath")
                link = gr.Textbox(label="…or a Google Drive / https link",
                                  placeholder="https://drive.google.com/file/d/…/view")
                vtt = gr.File(label="Zoom transcript (.vtt), optional — faster and more accurate",
                              file_types=[".vtt"], type="filepath")
                go = gr.Button("Grade this class", variant="primary")
                gr.Markdown("Takes ~1 minute with a .vtt, ~2 minutes without. Uploaded files are deleted "
                            "after grading.")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                with gr.Accordion("Evidence by pillar", open=False):
                    evidence = gr.Markdown()
                with gr.Accordion("Measured metrics", open=False):
                    metrics = gr.Markdown()
        go.click(run, [upload, link, vtt], [summary, evidence, metrics])
    with gr.Tab("Benchmark"):
        gr.Markdown(benchmark_md())
    with gr.Tab("How it works & privacy"):
        gr.Markdown(ABOUT)

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    user, pw = os.getenv("APP_USER"), os.getenv("APP_PASS")
    if os.getenv("SPACE_ID") and not (user and pw):
        raise SystemExit("APP_USER / APP_PASS secrets are required when running on Hugging Face.")
    # SSR crashes on Spaces (svelte-i18n locale error) and then fails the localhost check
    demo.launch(auth=(user, pw) if user and pw else None, max_file_size="2gb", ssr_mode=False)
