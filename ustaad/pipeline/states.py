"""Snapshots at still moments and the deterministic diff between consecutive snapshots.

This is the core robustness trick: SAM decides WHEN and WHAT changed (cheap, no LLM);
Glimmer only has to NAME the change.
"""
from __future__ import annotations

from collections import defaultdict

from ..schemas import Snapshot, StateDiff, StillMoment, TrackedObject


def snapshots(moments: list[StillMoment], tracks: dict[int, list[TrackedObject]], roles=("task",)) -> list[Snapshot]:
    return [Snapshot(k=m.k, frame=m.frame, objects=[o for o in tracks.get(m.frame.idx, []) if o.role in roles])
            for m in moments]


def _center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def diff(a: Snapshot, b: Snapshot, move_thresh: float, attach_iou: float, did: str) -> StateDiff:
    A = {o.obj_id: o for o in a.objects}
    B = {o.obj_id: o for o in b.objects}
    ca, cb = defaultdict(int), defaultdict(int)
    for o in a.objects:
        ca[o.concept] += 1
    for o in b.objects:
        cb[o.concept] += 1
    added = sorted({c for c in cb if cb[c] > ca.get(c, 0)})
    removed = sorted({c for c in ca if ca[c] > cb.get(c, 0)})
    moved = set()
    for k in A.keys() & B.keys():
        (x0, y0), (x1, y1) = _center(A[k].box), _center(B[k].box)
        if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > move_thresh:
            moved.add(A[k].concept)
    # objects whose ids changed but count stayed: compare per-concept centroid sets
    for c in set(ca) & set(cb):
        if c in moved or ca[c] != cb[c]:
            continue
        ra = sorted(_center(o.box) for o in a.objects if o.concept == c)
        rb = sorted(_center(o.box) for o in b.objects if o.concept == c)
        if any(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 > move_thresh for p, q in zip(ra, rb)):
            moved.add(c)
    attached, detached = set(), set()
    concepts = sorted(set(ca) & set(cb))
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i + 1:]:
            ia = max((_iou(p.box, q.box) for p in a.objects if p.concept == c1 for q in a.objects if q.concept == c2), default=0)
            ib = max((_iou(p.box, q.box) for p in b.objects if p.concept == c1 for q in b.objects if q.concept == c2), default=0)
            if ia < attach_iou <= ib:
                attached.add((c1, c2))
            elif ib < attach_iou <= ia:
                detached.add((c1, c2))
    d = StateDiff(id=did, before=a, after=b, added=added, removed=removed, moved=sorted(moved),
                  attached=[list(p) for p in sorted(attached)], detached=[list(p) for p in sorted(detached)])
    d.score = float(len(added) + len(removed) + len(moved) + len(attached) + len(detached))
    return d


def diffs(snaps: list[Snapshot], move_thresh: float, attach_iou: float, prefix: str = "d") -> list[StateDiff]:
    return [diff(snaps[i], snaps[i + 1], move_thresh, attach_iou, f"{prefix}{i + 1}") for i in range(len(snaps) - 1)]


def similarity(a: StateDiff, b: StateDiff) -> float:
    """Jaccard over (change type, concept) pairs."""
    sa = {(k, v) for k, vs in a.signature().items() for v in vs}
    sb = {(k, v) for k, vs in b.signature().items() for v in vs}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def type_overlap(a: StateDiff, b: StateDiff) -> bool:
    ta = {k for k, vs in a.signature().items() if vs}
    tb = {k for k, vs in b.signature().items() if vs}
    return bool(ta & tb)
