import numpy as np

from ustaad.pipeline.attempt import find_mismatches, match
from ustaad.pipeline.states import diff, similarity
from ustaad.perception.stillness import find_still_moments
from ustaad.schemas import FrameRef, Precondition, ProcedureGraph, Snapshot, Step, TrackedObject


def obj(c, i, x, y, w=0.05, h=0.05, role="task"):
    return TrackedObject(obj_id=f"{c}#{i}", concept=c, role=role, box=[x, y, w, h], area=w * h)


def snap(k, objs):
    return Snapshot(k=k, frame=FrameRef(idx=k, t=float(k)), objects=objs)


def test_diff_detects_move_add_remove():
    a = snap(0, [obj("nut", 0, .2, .2), obj("jack", 0, .8, .5), obj("wheel", 0, .3, .3, .2, .2)])
    b = snap(1, [obj("nut", 0, .7, .8), obj("jack", 0, .8, .5), obj("spanner", 0, .5, .5)])
    d = diff(a, b, 0.06, 0.15, "d1")
    assert d.moved == ["nut"] and d.added == ["spanner"] and d.removed == ["wheel"]
    assert d.score == 3


def test_similarity():
    a = snap(0, [obj("nut", 0, .2, .2)])
    b = snap(1, [obj("nut", 0, .7, .8)])
    d1, d2 = diff(a, b, .06, .15, "x"), diff(a, b, .06, .15, "y")
    assert similarity(d1, d2) == 1.0


def _graph_and_diffs():
    s0 = snap(0, [obj("nut", 0, .2, .2), obj("jack", 0, .8, .5)])
    s1 = snap(1, [obj("nut", 0, .7, .8), obj("jack", 0, .8, .5)])
    s2 = snap(2, [obj("nut", 0, .7, .8), obj("jack", 0, .3, .7)])
    m1, m2 = diff(s0, s1, .06, .15, "d1"), diff(s1, s2, .06, .15, "d2")
    mk = lambda i, d, title: Step(id=f"s{i}", diff_id=d.id, title=title, action="move", parts=[], end_state="",
                                  checkable=True, start=d.before.frame, end=d.after.frame)
    g = ProcedureGraph(title="t", steps=[mk(1, m1, "nuts"), mk(2, m2, "jack")],
                       preconditions=[Precondition(before="s1", after="s2", reason="r")])
    return g, {"d1": m1, "d2": m2}, (s0, s1, s2)


def test_order_violation_and_pass():
    g, md, _ = _graph_and_diffs()
    t0 = snap(0, [obj("nut", 0, .2, .2), obj("jack", 0, .8, .5)])
    t1 = snap(1, [obj("nut", 0, .2, .2), obj("jack", 0, .3, .7)])
    t2 = snap(2, [obj("nut", 0, .7, .8), obj("jack", 0, .3, .7)])
    wrong = [diff(t0, t1, .06, .15, "L1"), diff(t1, t2, .06, .15, "L2")]
    m = match(g, md, wrong, .45)
    mm = find_mismatches(g, md, wrong, m, .5)
    assert [x.type for x in mm] == ["order"]
    g.any_order = [["s1", "s2"]]
    assert find_mismatches(g, md, wrong, m, .5) == []


def test_skipped():
    g, md, (s0, s1, s2) = _graph_and_diffs()
    only_jack = [diff(s1, s2, .06, .15, "L1")]
    m = match(g, md, only_jack, .45)
    assert [x.type for x in find_mismatches(g, md, only_jack, m, .5)] == ["skipped"]


def test_still_moments_include_ends():
    refs = [FrameRef(idx=i, t=i / 3) for i in range(12)]
    e = np.array([0, 1, 1, 9, 9, 1, 1, 1, 9, 9, 1, 1], float)
    ms = find_still_moments(e, refs, 4.0, 2)
    idx = [m.frame.idx for m in ms]
    assert idx[0] == 0 and idx[-1] == 11 and len(idx) >= 3
