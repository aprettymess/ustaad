"""Deterministic stand-in for Glimmer so the whole pipeline runs without a GPU (tests, UI work)."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from ..schemas import EdgesOut, InventoryOut, JudgeOut, LeakReportOut, Precondition, StepOut


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")


class MockGlimmer:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or (lambda *a, **k: None)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def structured(self, system, content, model, strength=None, stage="llm"):
        self.usage["calls"] += 1
        t = _text(content)
        if model is InventoryOut:
            out = InventoryOut(task_title="Change a car tyre (synthetic)", task_concepts=["wheel", "lug nut", "jack"],
                               privacy_concepts=[], keep_rules=["tyre size marking"],
                               hide_rules=["hide faces", "hide licence plates"],
                               hazard={"refused": False, "reason": ""})
        elif model is StepOut:
            m = re.search(r"Tracked change: (.*)", t)
            change = m.group(1) if m else ""
            if "lug nut" in change:
                out = StepOut(title="Loosen and remove the lug nuts", action="remove", parts=["lug nut", "wheel"],
                              end_state="All lug nuts are off the wheel and in the tray", checkable=True,
                              tip="Loosen nuts in a star pattern")
            elif "jack" in change:
                out = StepOut(title="Place the jack under the car", action="move", parts=["jack"],
                              end_state="Jack is positioned under the jacking point", checkable=True)
            elif "wheel" in change:
                out = StepOut(title="Remove the wheel", action="remove", parts=["wheel"],
                              end_state="Wheel is off the hub", checkable=True)
            else:
                out = StepOut(title="Adjust parts", action="other", parts=[], end_state=change or "unclear",
                              checkable=False)
        elif model is EdgesOut:
            steps = dict(re.findall(r"^(s\d+): (.*)$", t, re.M))
            nut = next((k for k, v in steps.items() if "lug nut" in v), None)
            jack = next((k for k, v in steps.items() if "jack" in v), None)
            pre = [Precondition(before=nut, after=jack, reason="nuts must be loosened while the wheel is on the ground")] \
                if nut and jack else []
            out = EdgesOut(preconditions=pre)
        elif model is LeakReportOut:
            out = LeakReportOut(leaks=[])
        elif model is JudgeOut:
            out = JudgeOut(confirmed=True, explanation="You did this step out of order — loosen the nuts first.",
                           explanation_ur="پہلے نٹ ڈھیلے کریں، پھر جیک لگائیں۔")
        else:
            raise NotImplementedError(model)
        self.log(stage, "llm", f"{model.__name__} ok (mock)", output=out.model_dump())
        return out

    def chat(self, messages, tools=None, response_format=None, strength=None, max_tokens=None):
        """Agent loop: call `finish` once, then stop."""
        self.usage["calls"] += 1
        already = any(m.get("role") == "tool" for m in messages)
        if tools and not already and any(t["function"]["name"] == "finish" for t in tools):
            call = SimpleNamespace(id="call_0", function=SimpleNamespace(name="finish", arguments=json.dumps({"note": "mock ok"})))
            msg = SimpleNamespace(content="", tool_calls=[call])
        else:
            msg = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)
