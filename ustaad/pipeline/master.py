"""Master pipeline: video -> privacy-safe procedure card.

GPU choreography on 12 GB (Glimmer and SAM never resident together):
  inventory (Glimmer) -> track (SAM) -> audit (Glimmer [-> SAM re-track -> Glimmer]) -> steps/verify/edges (Glimmer)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from ..glimmer import get_glimmer
from ..glimmer.client import user_content
from ..glimmer.prompts import INVENTORY_SYSTEM
from ..glimmer.server import ensure_down, ensure_up
from ..perception.masks import MaskStore
from ..perception.sam import get_tracker
from ..perception.stillness import find_still_moments, motion_energy
from ..runctx import RunContext
from ..schemas import (FrameRef, InventoryOut, LeakFinding, ProcedureGraph, RedactionReceipt, StateDiff, Step,
                       StillMoment, TrackedObject)
from ..video import ingest, load_frame, sample_frames
from . import procedure as proc
from .leak_audit import run_audit
from .redact import render_sampled, render_video
from .states import diffs as make_diffs
from .states import snapshots

Tracks = dict[int, list[TrackedObject]]


class RefusedTask(RuntimeError):
    pass


def _privacy_keys(tracks: Tracks, exclude: set[str]) -> set[str]:
    return {o.obj_id for objs in tracks.values() for o in objs if o.role == "privacy"} - exclude


def render_outputs(ctx: RunContext, cfg, exclude: set[str] | None = None) -> RedactionReceipt:
    """(Re)render redacted + overlay videos. Also used by click-to-unblur."""
    refs = ctx.load("sample", list[FrameRef])
    tracks = ctx.load("tracks", Tracks)
    fixes = ctx.load("fixes") if (ctx.dir / "fixes.json").exists() else []
    exclude = set(exclude or [])
    keys = _privacy_keys(tracks, exclude)
    store = MaskStore(ctx.dir / "masks")
    n_red = render_video(ctx.dir / "source.mp4", ctx.dir / "redacted.mp4", refs, store, keys, fixes,
                         cfg.privacy, overlay_out=ctx.dir / "overlay.mp4")
    render_sampled(ctx.frames_dir, ctx.dir / "redacted", refs, store, keys, fixes, cfg.privacy)
    concepts: dict[str, set[str]] = {}
    low = []
    for r in refs:
        for o in tracks.get(r.idx, []):
            if o.role == "privacy" and o.obj_id in keys:
                concepts.setdefault(o.concept, set()).add(o.obj_id)
                if o.prob < 0.6:
                    low.append(r)
    leaks = ctx.load("leak_audit", list[LeakFinding]) if (ctx.dir / "leak_audit.json").exists() else []
    queue = {r.idx: r for r in low}
    for f in leaks:
        if f.verified is False:
            queue[f.frame.idx] = f.frame
    from ..video import probe
    receipt = RedactionReceipt(concepts={k: len(v) for k, v in concepts.items()},
                               frames_total=probe(ctx.dir / "source.mp4")["frames"], frames_with_redaction=n_red,
                               leaks=leaks, review_queue=sorted(queue.values(), key=lambda f: f.t),
                               excluded_obj_ids=sorted(exclude))
    ctx.save_json("receipt", receipt)
    return receipt


def run_master(video: str | Path, policy: str, cfg, run_id: str | None = None, force: bool = False,
               on_event=None) -> RunContext:
    ctx = RunContext(cfg.runs_dir, run_id, "master", force, on_event)
    src = ctx.dir / "source.mp4"
    if not src.exists():
        ingest(video, src)  # H.264, rotation applied, <=1080p
    ctx.save_json("meta", {"kind": "master", "policy": policy, "video": Path(video).name})
    G, gcfg = get_glimmer(cfg.glimmer, ctx.log), cfg.glimmer
    mx = gcfg.image_max_side

    refs = ctx.stage("sample", lambda: sample_frames(src, ctx.frames_dir, cfg.sampling.fps, cfg.sampling.max_side),
                     list[FrameRef])

    # 1. inventory + privacy plan + hazard check (Glimmer)
    def inventory():
        ensure_up(gcfg, ctx.log)
        picks = sorted({0, len(refs) // 2, len(refs) - 1})
        text = (f"User's privacy policy: \"{policy or 'hide anything that identifies people or our home'}\"\n"
                f"Default privacy list (already hidden): {', '.join(cfg.privacy.default_concepts)}\n"
                f"Hazard refusal list: {', '.join(cfg.hazards.refuse)}\n"
                f"Frames are from the start, middle and end of the video.")
        return G.structured(INVENTORY_SYSTEM, user_content(text, [load_frame(ctx.frames_dir, i) for i in picks],
                                                            [f"t={refs[i].t:.0f}s" for i in picks], mx),
                            InventoryOut, stage="inventory")
    inv = ctx.stage("inventory", inventory, InventoryOut)
    if inv.hazard.refused:
        ctx.save_json("status", {"state": "refused", "reason": inv.hazard.reason})
        raise RefusedTask(inv.hazard.reason)

    task = [c.lower() for c in inv.task_concepts]
    privacy = list(dict.fromkeys(c.lower() for c in cfg.privacy.default_concepts + inv.privacy_concepts if c.lower() not in task))
    concepts = [(c, "task") for c in task] + [("hand", "hand")] + [(c, "privacy") for c in privacy]
    store = MaskStore(ctx.dir / "masks")
    tracker = get_tracker(cfg.sam, ctx.log)

    # 2. one SAM session, one prompt per concept (Glimmer stopped to free VRAM)
    def track():
        ensure_down(gcfg, ctx.log)
        t = tracker.track(ctx.frames_dir, concepts, store)
        tracker.unload()
        return t
    tracks: Tracks = ctx.stage("tracks", track, Tracks)

    # 3. still moments -> snapshots -> diffs (deterministic)
    def moments():
        e = motion_energy(ctx.frames_dir, len(refs))
        return find_still_moments(e, refs, cfg.stillness.motion_thresh, cfg.stillness.min_still_frames,
                                  tracks, cfg.stillness.hand_area_max)
    ms = ctx.stage("moments", moments, list[StillMoment])
    diffs = ctx.stage("diffs", lambda: make_diffs(snapshots(ms, tracks), cfg.states.move_thresh,
                                                  cfg.states.attach_iou), list[StateDiff])

    # 4. privacy render on sampled frames, then agentic leak audit
    fixes: list[dict] = ctx.load("fixes") if (ctx.dir / "fixes.json").exists() else []
    render_sampled(ctx.frames_dir, ctx.dir / "redacted", refs, store, _privacy_keys(tracks, set()), fixes, cfg.privacy)

    def audit():
        ensure_up(gcfg, ctx.log)
        tracked_privacy = set(privacy)

        def rerender(only):
            render_sampled(ctx.frames_dir, ctx.dir / "redacted", refs, store, _privacy_keys(tracks, set()), fixes,
                           cfg.privacy, only=only)

        def retrack(new):
            ctx.log("leak_audit", "sam", f"re-tracking new privacy concepts: {new}")
            ensure_down(gcfg, ctx.log)
            extra = tracker.track(ctx.frames_dir, [(c, "privacy") for c in new], store)
            tracker.unload()
            for k, objs in extra.items():
                tracks.setdefault(k, []).extend(objs)
            ctx.save_json("tracks", tracks)
            ensure_up(gcfg, ctx.log)

        out = run_audit(G, refs, ctx.dir / "redacted", tracks, inv, cfg.leak_audit, rerender, retrack,
                        tracked_privacy, fixes, ctx.log, mx)
        ctx.save_json("fixes", fixes)
        return out
    ctx.stage("leak_audit", audit, list[LeakFinding])
    receipt = ctx.stage("receipt", lambda: render_outputs(ctx, cfg), RedactionReceipt)

    # 5. procedure graph (Glimmer)
    def steps():
        ensure_up(gcfg, ctx.log)
        return proc.name_steps(G, diffs, ctx.frames_dir, inv, cfg.procedure.min_diff_score, mx)
    draft = ctx.stage("steps_draft", steps, list[Step])
    verified = ctx.stage("steps", lambda: proc.verify_steps(G, draft, diffs, cfg.procedure.verifier_max_steps, ctx.log),
                         list[Step])
    graph = ctx.stage("procedure", lambda: proc.build_graph(inv.task_title, verified, proc.infer_edges(G, verified)),
                      ProcedureGraph)

    # 6. shareable card thumbnails come from REDACTED frames only
    card = ctx.dir / "card"
    card.mkdir(exist_ok=True)
    for s in graph.steps:
        for tag, fr in (("before", s.start), ("after", s.end)):
            img = cv2.imread(str(ctx.dir / "redacted" / f"{fr.idx:05d}.jpg"))
            if img is not None:
                cv2.imwrite(str(card / f"{s.id}_{tag}.jpg"), img)
    ctx.save_json("usage", getattr(G, "usage", {}))
    ctx.save_json("status", {"state": "done", "steps": len(graph.steps),
                             "privacy_objects": sum(receipt.concepts.values())})
    return ctx
