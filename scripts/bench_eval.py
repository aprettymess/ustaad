"""Benchmark: seeded-error detection + planted-PII redaction recall. Numbers for the README and the video.

Label files (bench/labels/*.json):

  attempts.json
  { "procedure": "master-...",              # run id of the master procedure
    "attempts": [ {"video": "bench/videos/pc_att1.mp4", "expected": ["order"]},      # seeded errors
                  {"video": "bench/videos/pc_att2.mp4", "expected": ["skipped"]},
                  {"video": "bench/videos/pc_att3.mp4", "expected": []} ] }           # clean attempt

  privacy.json
  { "runs": [ {"run": "master-...", "items": [ {"concept": "face", "t0": 0.0, "t1": 6.5},
                                               {"concept": "licence plate", "t0": 2.0, "t1": 30.0} ]} ] }

  python scripts/bench_eval.py attempts bench/labels/attempts.json
  python scripts/bench_eval.py privacy  bench/labels/privacy.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ustaad.config import load_config  # noqa: E402
from ustaad.runctx import RunContext  # noqa: E402

TYPES = ["skipped", "order", "wrong_part", "extra"]


def eval_attempts(path, cfg):
    from ustaad.pipeline.learner import run_attempt
    spec = json.loads(Path(path).read_text())
    tp, fp, fn = Counter(), Counter(), Counter()
    rows = []
    for a in spec["attempts"]:
        ctx = run_attempt(a["video"], spec["procedure"], cfg, run_id=a.get("run_id"))
        fb = ctx.load("feedback")
        pred = Counter(m["type"] for m in fb["mismatches"] if m.get("confirmed") is not False)
        exp = Counter(a["expected"])
        for t in TYPES:
            tp[t] += min(pred[t], exp[t])
            fp[t] += max(0, pred[t] - exp[t])
            fn[t] += max(0, exp[t] - pred[t])
        rows.append((Path(a["video"]).name, dict(exp), dict(pred)))
    print(f"{'attempt':28} {'expected':28} predicted")
    for n, e, p in rows:
        print(f"{n:28} {str(e):28} {p}")
    print(f"\n{'type':12} {'caught':>8} {'missed':>8} {'false alarms':>13} {'precision':>10} {'recall':>8}")
    for t in TYPES:
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else float("nan")
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else float("nan")
        print(f"{t:12} {tp[t]:>8} {fn[t]:>8} {fp[t]:>13} {p:>10.2f} {r:>8.2f}")


def eval_privacy(path, cfg):
    """Recall proxy: fraction of labelled sampled frames where a mask of that concept (or an audit box) was active."""
    spec = json.loads(Path(path).read_text())
    hit, tot = defaultdict(int), defaultdict(int)
    for run in spec["runs"]:
        ctx = RunContext(cfg.runs_dir, run["run"])
        refs = ctx.load("sample")
        tracks = ctx.load("tracks")
        fixes = ctx.load("fixes") if (ctx.dir / "fixes.json").exists() else []
        for it in run["items"]:
            for r in refs:
                if not (it["t0"] <= r["t"] <= it["t1"]):
                    continue
                tot[it["concept"]] += 1
                masked = any(o["concept"] == it["concept"] and o["role"] == "privacy" for o in tracks.get(str(r["idx"]), []))
                boxed = any(f["t0"] <= r["t"] <= f["t1"] for f in fixes)
                hit[it["concept"]] += masked or boxed
    print(f"{'concept':18} {'frames covered':>15} {'recall':>8}")
    for c in tot:
        print(f"{c:18} {hit[c]:>7}/{tot[c]:<7} {hit[c] / tot[c]:>8.2%}")
    print("\nAlso eyeball redacted.mp4 frame-by-frame for anything the labels missed.")


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    cfg = load_config()
    {"attempts": eval_attempts, "privacy": eval_privacy}[cmd](path, cfg)
