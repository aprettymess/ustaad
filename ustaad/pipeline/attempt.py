"""Compare a learner's attempt to the master's procedure graph. Camera-verifiable claims only."""
from __future__ import annotations

from pathlib import Path

import cv2

from ..glimmer.client import user_content
from ..glimmer.prompts import JUDGE_SYSTEM
from ..schemas import Feedback, JudgeOut, Mismatch, ProcedureGraph, StateDiff
from ..video import load_frame, side_by_side
from .states import similarity, type_overlap


def match(graph: ProcedureGraph, master_diffs: dict[str, StateDiff], learner: list[StateDiff],
          thresh: float) -> dict[str, str]:
    pairs = []
    for s in graph.steps:
        md = master_diffs.get(s.diff_id)
        if md is None:
            continue
        for ld in learner:
            sc = similarity(md, ld)
            if sc >= thresh:
                pairs.append((sc, s.id, ld.id))
    pairs.sort(reverse=True)
    used_s, used_l, out = set(), set(), {}
    for sc, sid, lid in pairs:
        if sid not in used_s and lid not in used_l:
            out[sid] = lid
            used_s.add(sid)
            used_l.add(lid)
    return out


def find_mismatches(graph: ProcedureGraph, master_diffs: dict[str, StateDiff], learner: list[StateDiff],
                    matched: dict[str, str], min_score: float) -> list[Mismatch]:
    lpos = {d.id: i for i, d in enumerate(learner)}
    lby = {d.id: d for d in learner}
    sby = {s.id: s for s in graph.steps}
    any_order = [set(g) for g in graph.any_order]
    out: list[Mismatch] = []
    unmatched_steps = [s for s in graph.steps if s.id not in matched and s.checkable]
    unmatched_l = [d for d in learner if d.id not in matched.values() and d.score >= min_score]
    # wrong part: same kind of change, different parts
    for s in list(unmatched_steps):
        md = master_diffs[s.diff_id]
        cand = next((d for d in unmatched_l if type_overlap(md, d)), None)
        if cand:
            out.append(Mismatch(type="wrong_part", step_id=s.id,
                                detail=f"Expected '{s.title}' ({md.summary()}), saw {cand.summary()}",
                                master_frame=s.end, learner_frame=cand.after.frame))
            unmatched_steps.remove(s)
            unmatched_l.remove(cand)
    for s in unmatched_steps:
        out.append(Mismatch(type="skipped", step_id=s.id, detail=f"No evidence of: {s.title}", master_frame=s.end))
    for p in graph.preconditions:
        if any(p.before in g and p.after in g for g in any_order):
            continue
        if p.before in matched and p.after in matched and lpos[matched[p.after]] < lpos[matched[p.before]]:
            a, b = sby[p.after], sby[p.before]
            out.append(Mismatch(type="order", step_id=p.after,
                                detail=f"'{a.title}' was done before '{b.title}' ({p.reason})",
                                master_frame=b.end, learner_frame=lby[matched[p.after]].after.frame))
    for d in unmatched_l:
        out.append(Mismatch(type="extra", detail=f"Unexpected change: {d.summary()}", learner_frame=d.after.frame))
    return out


def judge(glimmer, mismatches: list[Mismatch], master_frames: Path, learner_frames: Path, out_dir: Path,
          graph: ProcedureGraph, image_max_side: int = 768) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(mismatches):
        imgs, labels = [], []
        if m.master_frame:
            imgs.append(load_frame(master_frames, m.master_frame.idx))
            labels.append("MASTER (what it should look like):")
        if m.learner_frame:
            imgs.append(load_frame(learner_frames, m.learner_frame.idx))
            labels.append("LEARNER:")
        if len(imgs) == 2:
            cv2.imwrite(str(out_dir / f"m{i}.jpg"), side_by_side(imgs[0], imgs[1]))
        elif imgs:
            cv2.imwrite(str(out_dir / f"m{i}.jpg"), imgs[0])
        if glimmer is None:
            continue
        steps = "; ".join(f"{s.id}: {s.title}" for s in graph.steps)
        r = glimmer.structured(JUDGE_SYSTEM, user_content(
            f"Job: {graph.title}\nSteps: {steps}\nPossible mistake ({m.type}): {m.detail}", imgs, labels, image_max_side),
            JudgeOut, stage="judge")
        m.confirmed, m.explanation, m.explanation_ur = r.confirmed, r.explanation, r.explanation_ur


def feedback(procedure_id: str, graph: ProcedureGraph, matched: dict[str, str], mismatches: list[Mismatch]) -> Feedback:
    failing = [m for m in mismatches if m.confirmed is not False and m.type != "extra"]
    return Feedback(procedure_id=procedure_id, matched=matched, mismatches=mismatches, passed=not failing,
                    not_checked_tips=[f"{s.title}: {s.tip}" for s in graph.steps if s.tip])
