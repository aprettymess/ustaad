"""Day-0 go/no-go tests on REAL models. Run each, write the result in docs/spike-results.md.

  python scripts/spikes.py sam     clip.mp4 --concepts "wheel,lug nut,jack,hand,face,licence plate"
  python scripts/spikes.py diff    before.jpg after.jpg --parts "wheel,lug nut,jack"
  python scripts/spikes.py diff    clip.mp4 --at 3.5 8.0 --parts "wheel,lug nut,jack"    (frames from a video)
  python scripts/spikes.py leak    redacted_frame.jpg
  python scripts/spikes.py urdu
  python scripts/spikes.py time    clip.mp4 --policy "hide faces and the plate"          (full master run, timings)

Pass bars (from concept-ustaad.md):
  sam : every concept found; distinct ids per concept ~= real count (ID swaps < 2 per minute)
  diff: >= 3 of 4 before/after pairs named correctly (action + part)
  leak: the planted leak is reported with a box that overlaps it
  urdu: a native speaker rates >= 4/5 sentences as clear
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ustaad.config import load_config  # noqa: E402
from ustaad.glimmer import get_glimmer  # noqa: E402
from ustaad.glimmer.client import user_content  # noqa: E402
from ustaad.glimmer.prompts import LEAK_SYSTEM, STEP_SYSTEM  # noqa: E402
from ustaad.perception.masks import MaskStore  # noqa: E402
from ustaad.perception.sam import get_tracker  # noqa: E402
from ustaad.schemas import LeakReportOut, StepOut  # noqa: E402
from ustaad.video import sample_frames  # noqa: E402


def frame_at(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {t}s")
    return f


def sam_spike(a, cfg):
    tmp = Path(tempfile.mkdtemp(prefix="ustaad_sam_"))
    refs = sample_frames(a.video, tmp / "frames", cfg.sampling.fps, cfg.sampling.max_side)
    concepts = [(c.strip(), "hand" if c.strip() == "hand" else "task") for c in a.concepts.split(",")]
    t0 = time.time()
    tracks = get_tracker(cfg.sam).track(tmp / "frames", concepts, MaskStore(tmp / "masks"))
    dt = time.time() - t0
    ids, per_frame = defaultdict(set), defaultdict(list)
    for idx, objs in tracks.items():
        cnt = defaultdict(int)
        for o in objs:
            ids[o.concept].add(o.obj_id)
            cnt[o.concept] += 1
        for c, n in cnt.items():
            per_frame[c].append(n)
    dur = refs[-1].t or 1
    print(f"\n{len(refs)} frames ({dur:.0f}s video) tracked in {dt:.0f}s  ({dt / max(dur, 1):.1f}s per video-second)")
    print(f"{'concept':18} {'distinct ids':>12} {'max at once':>12} {'frames seen':>12}")
    for c, _ in concepts:
        seen = len(per_frame[c])
        print(f"{c:18} {len(ids[c]):>12} {max(per_frame[c], default=0):>12} {seen:>9}/{len(refs)}")
    print("\nPASS if every concept is seen and distinct ids ~= max at once (more ids = identity swaps).")
    # quick visual check
    out = tmp / "overlay.mp4"
    store = MaskStore(tmp / "masks")
    first = cv2.imread(str(tmp / "frames" / "00000.jpg"))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), cfg.sampling.fps, (first.shape[1], first.shape[0]))
    rng = {}
    import numpy as np
    for r in refs:
        img = cv2.imread(str(tmp / "frames" / f"{r.idx:05d}.jpg"))
        for k, m in store.get(r.idx).items():
            col = rng.setdefault(k, np.random.randint(60, 255, 3))
            img[m] = (0.5 * img[m] + 0.5 * col).astype("uint8")
        vw.write(img)
    vw.release()
    print(f"overlay video: {out}")


def diff_spike(a, cfg):
    G = get_glimmer(cfg.glimmer)
    if a.inputs[0].lower().endswith((".mp4", ".mov", ".mkv")):
        before, after = frame_at(a.inputs[0], a.at[0]), frame_at(a.inputs[0], a.at[1])
    else:
        before, after = cv2.imread(a.inputs[0]), cv2.imread(a.inputs[1])
    t0 = time.time()
    out = G.structured(STEP_SYSTEM, user_content(
        f"Inventory part names: {a.parts}\nTracked change: {a.change or 'unknown'}", [before, after],
        ["BEFORE:", "AFTER:"], cfg.glimmer.image_max_side), StepOut)
    print(out.model_dump_json(indent=2))
    print(f"{time.time() - t0:.1f}s, usage={G.usage}")


def leak_spike(a, cfg):
    G = get_glimmer(cfg.glimmer)
    t0 = time.time()
    rep = G.structured(LEAK_SYSTEM, user_content("Keep rules: none. List remaining leaks.", [cv2.imread(a.image)],
                                                 max_side=cfg.glimmer.image_max_side), LeakReportOut)
    print(rep.model_dump_json(indent=2))
    print(f"{time.time() - t0:.1f}s")


def urdu_spike(a, cfg):
    G = get_glimmer(cfg.glimmer)
    sents = ["Loosen the wheel nuts before you lift the car with the jack.",
             "You skipped putting the filter back before closing the cover.",
             "This part goes on the left side, not the right.",
             "Good — this step matches the master's video.",
             "Tighten the nuts in a star pattern; a camera cannot check this, so take care."]
    r = G.chat([{"role": "user", "content": "Translate each line to simple everyday Urdu (Urdu script). "
                                            "Return one line per sentence, numbered.\n" + "\n".join(
                    f"{i + 1}. {s}" for i, s in enumerate(sents))}], strength="low")
    print(r.choices[0].message.content)
    print("\nAsk a native speaker to rate each line 1-5 for clarity.")


def time_spike(a, cfg):
    from ustaad.pipeline.master import run_master
    ctx = run_master(a.video, a.policy, cfg, force=True)
    print("timings (s):", ctx.timings, "\nglimmer usage:", ctx.load("usage"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sam"); s.add_argument("video"); s.add_argument("--concepts", required=True)
    d = sub.add_parser("diff"); d.add_argument("inputs", nargs="+"); d.add_argument("--at", nargs=2, type=float)
    d.add_argument("--parts", required=True); d.add_argument("--change", default="")
    l = sub.add_parser("leak"); l.add_argument("image")
    sub.add_parser("urdu")
    t = sub.add_parser("time"); t.add_argument("video"); t.add_argument("--policy", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"sam": sam_spike, "diff": diff_spike, "leak": leak_spike, "urdu": urdu_spike, "time": time_spike}[a.cmd](a, cfg)
