"""Agentic leak audit: Glimmer inspects REDACTED frames, we apply its fixes, then it re-checks.

Fix actions: an immediate box mask over +/- fix_window_s (always, for safety) and, if the leak is a
concept SAM wasn't tracking, a re-track of that concept across the whole video (batched per round).
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..glimmer.client import user_content
from ..glimmer.prompts import LEAK_SYSTEM
from ..schemas import FrameRef, InventoryOut, LeakFinding, LeakReportOut, TrackedObject


def pick_audit_frames(refs: list[FrameRef], tracks: dict[int, list[TrackedObject]], n: int) -> list[int]:
    """Evenly spaced frames + 'risky' frames where the set of privacy objects changes (appear/vanish)."""
    even = set(np.linspace(0, len(refs) - 1, num=min(n, len(refs))).round().astype(int).tolist())
    risky, prev = set(), None
    for r in refs:
        cur = frozenset(o.obj_id for o in tracks.get(r.idx, []) if o.role == "privacy")
        if prev is not None and cur != prev:
            risky.update({r.idx, min(r.idx + 1, len(refs) - 1)})
        prev = cur
    picked = sorted(even | risky)
    if len(picked) > n * 2:  # keep budget bounded
        picked = sorted(even | set(sorted(risky)[: n]))
    return picked


def _overlaps(a, b) -> bool:
    return not (a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1])


def run_audit(glimmer, refs: list[FrameRef], redacted_dir, tracks, inventory: InventoryOut, cfg,
              rerender: Callable[[list[int]], None], retrack: Callable[[list[str]], None],
              tracked_privacy: set[str], fixes: list[dict], log, image_max_side: int = 768) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    frames = pick_audit_frames(refs, tracks, cfg.frames)
    keep = "; ".join(inventory.keep_rules) or "none"
    for rnd in range(1, cfg.max_rounds + 1):
        new_concepts: set[str] = set()
        round_findings: list[LeakFinding] = []
        for idx in frames:
            img = cv2.imread(str(redacted_dir / f"{idx:05d}.jpg"))
            rep = glimmer.structured(LEAK_SYSTEM, user_content(
                f"Keep rules (must stay visible): {keep}\nFrame time: {refs[idx].t:.1f}s. List remaining leaks.",
                [img], max_side=image_max_side), LeakReportOut, stage="leak_audit")
            for item in rep.leaks:
                fix = "box"
                c = item.concept.strip().lower()
                if c and c not in tracked_privacy and c not in {k.lower() for k in inventory.keep_rules}:
                    new_concepts.add(c)
                    fix = "retrack"
                fixes.append({"t0": max(0.0, refs[idx].t - cfg.fix_window_s), "t1": refs[idx].t + cfg.fix_window_s,
                              "box": item.box, "reason": item.description})
                round_findings.append(LeakFinding(round=rnd, frame=refs[idx], item=item, fix=fix))
                log("leak_audit", "fix", f"leak @ {refs[idx].t:.1f}s: {item.description} -> {fix}", box=item.box)
        if not round_findings:
            log("leak_audit", "info", f"round {rnd}: clean")
            break
        if new_concepts:
            retrack(sorted(new_concepts))
            tracked_privacy |= new_concepts
        leak_frames = sorted({f.frame.idx for f in round_findings})
        rerender(leak_frames)
        # verify: re-inspect only the frames that had leaks
        for idx in leak_frames:
            img = cv2.imread(str(redacted_dir / f"{idx:05d}.jpg"))
            rep = glimmer.structured(LEAK_SYSTEM, user_content(
                f"Keep rules: {keep}\nRe-check after fixes. List remaining leaks.", [img], max_side=image_max_side),
                LeakReportOut, stage="leak_audit")
            for f in round_findings:
                if f.frame.idx == idx:
                    f.verified = not any(_overlaps(f.item.box, l.box) for l in rep.leaks)
        findings += round_findings
        frames = leak_frames  # next round focuses on frames that had problems
    return findings
