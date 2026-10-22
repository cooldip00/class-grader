"""Run the grader over the labelled dataset LOCALLY and write benchmark_results.json.

The dataset is confidential and never leaves this machine except as pseudonymised
transcripts sent to the LLM. The public results file holds only batch IDs, labels,
predictions and scores - no quotes, names or evidence text.

usage: python scripts/benchmark.py <dataset_root> [--only B482] [--video] [--rejudge]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import find_dotenv, load_dotenv  # noqa: E402

from grader import judge as J  # noqa: E402
from grader import score as S  # noqa: E402
from grader import transcript as T  # noqa: E402

CACHE = ROOT / ".cache" / "benchmark"


def pair_key(name: str) -> str:
    """Zoom names the .vtt differently from the .mp4 (no date, different spacing),
    so pair on the batch ID, or 'demo <language>' for the demo folder."""
    toks = Path(name).stem.lower().replace("-", " ").split()
    return " ".join(toks[:2]) if toks[0] == "demo" else toks[0]


def find_pairs(root: Path) -> list[dict]:
    pairs = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        vtts = {pair_key(f.name): f for f in folder.glob("*.vtt")}
        for mp4 in sorted(folder.glob("*.mp4")):
            key = pair_key(mp4.name)
            pairs.append({
                "id": f"{folder.name}/{key.upper() if key[0] == 'b' else key}",
                "batch": key.upper() if key[0] == "b" else key,
                "label": S.LABELS.get(folder.name),  # None for demo recordings
                "mp4": mp4, "vtt": vtts.get(key),
            })
    return pairs


def run_one(p: dict, client, rejudge: bool, video: bool) -> dict:
    slug = p["id"].replace("/", "__").replace(" ", "_")
    cache_file = CACHE / J.PROMPT_VERSION / f"{slug}.json"
    tr = T.load(p["vtt"].read_text(encoding="utf-8"))
    m = T.metrics(tr)
    media = None
    if video:
        from grader import media as M
        media = M.analyse(str(p["mp4"]), tr)
    fresh = not (cache_file.exists() and not rejudge)
    if not fresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        jd, usage = J.Judgement.model_validate(cached["judgement"]), cached["usage"]
    else:
        jd, usage = J.judge(T.render(tr), m, client=client)  # audio is informational, not judged
        cache_file.write_text(json.dumps({"judgement": jd.model_dump(), "usage": usage}, indent=1),
                              encoding="utf-8")
    pillars = {k: getattr(jd, k).score for k in J.PILLARS}
    visual = (media or {}).get("visual_score")
    result = S.grade(pillars, m.get("english_ratio"), jd.confidence, visual)
    pillars["visual_compliance"] = visual
    return {"id": p["id"], "batch": p["batch"], "label": p["label"], "predicted": result["grade"],
            "llm_holistic": jd.holistic_grade, "weighted_score": result["weighted_score"],
            "confidence": result["confidence"], "pillars": pillars,
            "duration_min": m.get("duration_min"), "english_ratio": m.get("english_ratio"),
            "visual": {k: v for k, v in (media or {}).items() if k not in ("audio", "timeline")} or None,
            "tokens": usage if fresh else {"input_tokens": 0, "output_tokens": 0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--only", help="substring of the id to run")
    ap.add_argument("--video", action="store_true", help="also run video/audio analysis")
    ap.add_argument("--rejudge", action="store_true", help="ignore cached LLM judgements")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    (CACHE / J.PROMPT_VERSION).mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(max_retries=6)
    pairs = [p for p in find_pairs(args.root) if p["vtt"] and (not args.only or args.only in p["id"])]

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda p: run_one(p, client, args.rejudge, args.video), pairs))

    order = {g: i for i, g in enumerate(S.GRADES)}
    labelled = [r for r in rows if r["label"]]
    exact = sum(r["predicted"] == r["label"] for r in labelled)
    within1 = sum(abs(order[r["predicted"]] - order[r["label"]]) <= 1 for r in labelled)
    holistic = sum(r["llm_holistic"] == r["label"] for r in labelled)

    print(f"{'id':28} {'actual':10} {'pred':10} {'llm':10} {'score':>5}  pillars")
    for r in rows:
        ps = " ".join(f"{v if v is not None else '-'}" for v in r["pillars"].values())
        print(f"{r['id']:28} {str(r['label']):10} {r['predicted']:10} {r['llm_holistic']:10} "
              f"{r['weighted_score']:>5}  {ps}")
    if labelled:
        print(f"\nexact {exact}/{len(labelled)} | within one grade {within1}/{len(labelled)} | "
              f"LLM holistic exact {holistic}/{len(labelled)}")
    tok = sum(r["tokens"]["input_tokens"] for r in rows), sum(r["tokens"]["output_tokens"] for r in rows)
    print(f"tokens in/out: {tok[0]}/{tok[1]}  (~${tok[0] * 2e-6 + tok[1] * 10e-6:.2f} on Sonnet 5.5)")

    if not args.only:
        public = [{k: v for k, v in r.items() if k != "tokens"} for r in rows]
        summary = {"prompt_version": J.PROMPT_VERSION, "model": J.MODEL,
                   "n_labelled": len(labelled), "exact": exact, "within_one": within1,
                   "llm_holistic_exact": holistic}
        (ROOT / "benchmark_results.json").write_text(
            json.dumps({"summary": summary, "rows": public}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
