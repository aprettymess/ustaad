"""State diffs -> named steps (Glimmer vision) -> verifier agent (tools) -> preconditions (graph)."""
from __future__ import annotations

from pathlib import Path

from ..glimmer.agent import Tool, run_agent
from ..glimmer.client import user_content
from ..glimmer.prompts import EDGES_SYSTEM, STEP_SYSTEM, VERIFIER_SYSTEM
from ..schemas import EdgesOut, InventoryOut, ProcedureGraph, StateDiff, Step, StepOut
from ..video import load_frame


def name_steps(glimmer, diffs: list[StateDiff], frames_dir: Path, inv: InventoryOut, min_score: float,
               image_max_side: int = 768) -> list[Step]:
    steps: list[Step] = []
    for d in diffs:
        if d.score < min_score:
            continue
        before, after = load_frame(frames_dir, d.before.frame.idx), load_frame(frames_dir, d.after.frame.idx)
        text = (f"Job: {inv.task_title}\nInventory part names: {', '.join(inv.task_concepts)}\n"
                f"Tracked change: {d.summary()}\nBEFORE is at {d.before.frame.t:.1f}s, AFTER at {d.after.frame.t:.1f}s.")
        out = glimmer.structured(STEP_SYSTEM, user_content(text, [before, after], ["BEFORE:", "AFTER:"], image_max_side),
                                 StepOut, stage="steps")
        steps.append(Step(**out.model_dump(), id=f"s{len(steps) + 1}", diff_id=d.id,
                          start=d.before.frame, end=d.after.frame))
    return steps


def verify_steps(glimmer, steps: list[Step], diffs: list[StateDiff], max_steps: int, log) -> list[Step]:
    by_id = {s.id: s for s in steps}
    dmap = {d.id: d for d in diffs}

    def list_steps():
        return [{"id": s.id, "title": s.title, "action": s.action, "parts": s.parts,
                 "checkable": s.checkable, "t": [s.start.t, s.end.t]} for s in by_id.values()]

    def get_evidence(step_id: str):
        s = by_id.get(step_id)
        if not s:
            return {"error": f"no step {step_id}", "hint": "call list_steps"}
        d = dmap[s.diff_id]
        return {"step": step_id, "tracked_change": d.summary(), "change_strength": d.score,
                "before_t": d.before.frame.t, "after_t": d.after.frame.t, "end_state": s.end_state}

    def merge_steps(keep_id: str, drop_id: str, new_title: str):
        if keep_id not in by_id or drop_id not in by_id:
            return {"error": "unknown step id", "hint": "call list_steps"}
        k, d = by_id[keep_id], by_id.pop(drop_id)
        k.title = new_title
        k.parts = sorted(set(k.parts) | set(d.parts))
        k.start, k.end = min(k.start, d.start, key=lambda f: f.t), max(k.end, d.end, key=lambda f: f.t)
        return {"ok": True, "steps": len(by_id)}

    def delete_step(step_id: str, reason: str):
        if by_id.pop(step_id, None) is None:
            return {"error": f"no step {step_id}"}
        return {"ok": True, "reason": reason}

    def rename_step(step_id: str, title: str):
        if step_id not in by_id:
            return {"error": f"no step {step_id}"}
        by_id[step_id].title = title
        return {"ok": True}

    def set_checkable(step_id: str, checkable: bool, reason: str):
        if step_id not in by_id:
            return {"error": f"no step {step_id}"}
        by_id[step_id].checkable = bool(checkable)
        return {"ok": True}

    def finish(note: str = ""):
        return {"ok": True, "note": note}

    S, B = {"type": "string"}, {"type": "boolean"}
    tools = [
        Tool("list_steps", "List the current draft steps.", {"type": "object", "properties": {}}, list_steps),
        Tool("get_evidence", "Show the tracker evidence behind a step.",
             {"type": "object", "properties": {"step_id": S}, "required": ["step_id"]}, get_evidence),
        Tool("merge_steps", "Merge two steps that describe the same action.",
             {"type": "object", "properties": {"keep_id": S, "drop_id": S, "new_title": S},
              "required": ["keep_id", "drop_id", "new_title"]}, merge_steps),
        Tool("delete_step", "Delete a step that has no real evidence.",
             {"type": "object", "properties": {"step_id": S, "reason": S}, "required": ["step_id", "reason"]}, delete_step),
        Tool("rename_step", "Give a step a clearer imperative title.",
             {"type": "object", "properties": {"step_id": S, "title": S}, "required": ["step_id", "title"]}, rename_step),
        Tool("set_checkable", "Mark whether a camera can verify this step.",
             {"type": "object", "properties": {"step_id": S, "checkable": B, "reason": S},
              "required": ["step_id", "checkable", "reason"]}, set_checkable),
        Tool("finish", "Call when the step list is correct.", {"type": "object", "properties": {"note": S}}, finish),
    ]
    listing = "\n".join(f"{s.id}: {s.title} [{s.action}; parts={s.parts}; checkable={s.checkable}]" for s in steps)
    run_agent(glimmer, VERIFIER_SYSTEM, f"Draft steps:\n{listing}", tools, max_steps=max_steps, stage="verifier", log=log)
    return sorted(by_id.values(), key=lambda s: s.start.t)


def infer_edges(glimmer, steps: list[Step]) -> EdgesOut:
    listing = "\n".join(f"{s.id}: {s.title} (parts: {', '.join(s.parts)})" for s in steps)
    out = glimmer.structured(EDGES_SYSTEM, f"Steps in the order the master did them:\n{listing}", EdgesOut,
                             stage="edges")
    ids = {s.id for s in steps}
    out.preconditions = [p for p in out.preconditions if p.before in ids and p.after in ids and p.before != p.after]
    return out


def build_graph(title: str, steps: list[Step], edges: EdgesOut) -> ProcedureGraph:
    return ProcedureGraph(title=title, steps=steps, preconditions=edges.preconditions)
