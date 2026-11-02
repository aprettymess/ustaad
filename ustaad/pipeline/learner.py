"""Learner pipeline: attempt video + master procedure -> private feedback with side-by-side evidence."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..glimmer import get_glimmer
from ..glimmer.server import ensure_down, ensure_up
from ..perception.masks import MaskStore
from ..perception.sam import get_tracker
from ..perception.stillness import find_still_moments, motion_energy
from ..runctx import RunContext
from ..schemas import Feedback, FrameRef, InventoryOut, ProcedureGraph, StateDiff, StillMoment, TrackedObject
from ..video import ingest, sample_frames
from . import attempt as att
from .states import diffs as make_diffs
from .states import snapshots

Tracks = dict[int, list[TrackedObject]]


def run_attempt(video: str | Path, procedure_id: str, cfg, run_id: str | None = None, force: bool = False,
                on_event=None) -> RunContext:
    master = RunContext(cfg.runs_dir, procedure_id, "master")
    graph = master.load("procedure", ProcedureGraph)
    inv = master.load("inventory", InventoryOut)
    mdiffs = {d.id: d for d in master.load("diffs", list[StateDiff])}

    ctx = RunContext(cfg.runs_dir, run_id, "attempt", force, on_event)
    src = ctx.dir / "source.mp4"
    if not src.exists():
        ingest(video, src)  # H.264, rotation applied, <=1080p
    ctx.save_json("meta", {"kind": "attempt", "procedure": procedure_id, "video": Path(video).name})
    gcfg = cfg.glimmer

    refs = ctx.stage("sample", lambda: sample_frames(src, ctx.frames_dir, cfg.sampling.fps, cfg.sampling.max_side),
                     list[FrameRef])
    concepts = [(c.lower(), "task") for c in inv.task_concepts] + [("hand", "hand")]
    tracker = get_tracker(cfg.sam, ctx.log)

    def track():
        ensure_down(gcfg, ctx.log)
        t = tracker.track(ctx.frames_dir, concepts, MaskStore(ctx.dir / "masks"))
        tracker.unload()
        return t
    tracks: Tracks = ctx.stage("tracks", track, Tracks)
    ms = ctx.stage("moments", lambda: find_still_moments(motion_energy(ctx.frames_dir, len(refs)), refs,
                                                         cfg.stillness.motion_thresh, cfg.stillness.min_still_frames,
                                                         tracks, cfg.stillness.hand_area_max), list[StillMoment])
    ldiffs = ctx.stage("diffs", lambda: make_diffs(snapshots(ms, tracks), cfg.states.move_thresh,
                                                   cfg.states.attach_iou, prefix="L"), list[StateDiff])

    def compare():
        matched = att.match(graph, mdiffs, ldiffs, cfg.attempt.match_thresh)
        mism = att.find_mismatches(graph, mdiffs, ldiffs, matched, cfg.procedure.min_diff_score)
        ctx.log("compare", "info", f"matched {len(matched)}/{len(graph.steps)} steps, {len(mism)} candidate mismatch(es)")
        G = None
        if cfg.attempt.judge_mismatches and mism:
            ensure_up(gcfg, ctx.log)
            G = get_glimmer(gcfg, ctx.log)
        # learners only ever see the master's REDACTED frames (the master video is the shared one)
        att.judge(G, mism, master.dir / "redacted", ctx.frames_dir, ctx.dir / "evidence", graph, gcfg.image_max_side)
        return att.feedback(procedure_id, graph, matched, mism)
    fb = ctx.stage("feedback", compare, Feedback)
    ctx.save_json("status", {"state": "done", "passed": fb.passed, "mismatches": len(fb.mismatches)})
    return ctx
