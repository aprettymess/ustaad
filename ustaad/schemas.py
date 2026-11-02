"""All data that flows between stages. Glimmer structured outputs use the *Out models."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Box = list[float]  # [x, y, w, h], normalized 0-1


# ---------- Glimmer structured outputs ----------

class HazardCheck(BaseModel):
    refused: bool = Field(description="true if the task involves any hazard on the refusal list")
    reason: str = Field(default="", description="short reason if refused")


class InventoryOut(BaseModel):
    task_title: str = Field(description="short name of the job being done, e.g. 'Change a car tyre'")
    task_concepts: list[str] = Field(description="physical parts and tools used in the job, short singular nouns SAM can segment, e.g. 'wheel', 'lug nut', 'jack', 'wrench'")
    privacy_concepts: list[str] = Field(description="extra things visible that are private and must be hidden, beyond the defaults, e.g. 'letter', 'calendar with names'")
    keep_rules: list[str] = Field(description="task-relevant text or objects that must stay visible, e.g. 'tyre size marking', 'appliance model number'")
    hide_rules: list[str] = Field(description="the user's privacy policy restated as concrete rules")
    hazard: HazardCheck


class StepOut(BaseModel):
    title: str = Field(description="imperative step title, max 8 words, e.g. 'Loosen the lug nuts'")
    action: Literal["add", "remove", "move", "attach", "detach", "adjust", "inspect", "other"]
    parts: list[str] = Field(description="task concepts involved, using the inventory names")
    end_state: str = Field(description="what is visibly true after this step")
    checkable: bool = Field(description="true only if a camera can verify this step from the before/after frames")
    tip: Optional[str] = Field(default=None, description="master's tip that a camera cannot verify (torque, feel, sound), else null")


class Precondition(BaseModel):
    before: str = Field(description="step id that must be completed first")
    after: str = Field(description="step id that depends on it")
    reason: str


class EdgesOut(BaseModel):
    preconditions: list[Precondition]


class LeakItem(BaseModel):
    description: str = Field(description="what private thing is still visible")
    box: list[float] = Field(description="[x, y, w, h] normalized 0-1 around the leak")
    concept: str = Field(description="short noun phrase SAM could track for it, e.g. 'licence plate'")


class LeakReportOut(BaseModel):
    leaks: list[LeakItem] = Field(description="empty if nothing private is visible")


class JudgeOut(BaseModel):
    confirmed: bool = Field(description="true if the learner really made this mistake")
    explanation: str = Field(description="one or two plain sentences to the learner, kind and specific")
    explanation_ur: Optional[str] = Field(default=None, description="same explanation in Urdu, or null")


# ---------- pipeline artifacts ----------

class FrameRef(BaseModel):
    idx: int          # index in sampled frames
    t: float          # seconds in source video


class TrackedObject(BaseModel):
    obj_id: str       # "<concept>#<n>"
    concept: str
    role: Literal["task", "privacy", "hand"]
    box: Box
    area: float       # mask area / frame area
    prob: float = 1.0


class StillMoment(BaseModel):
    k: int
    frame: FrameRef


class Snapshot(BaseModel):
    k: int
    frame: FrameRef
    objects: list[TrackedObject]


class StateDiff(BaseModel):
    id: str
    before: Snapshot
    after: Snapshot
    added: list[str] = []
    removed: list[str] = []
    moved: list[str] = []
    attached: list[list[str]] = []
    detached: list[list[str]] = []
    score: float = 0.0

    def signature(self) -> dict[str, set[str]]:
        return {
            "added": set(self.added), "removed": set(self.removed), "moved": set(self.moved),
            "attached": {"+".join(sorted(p)) for p in self.attached},
            "detached": {"+".join(sorted(p)) for p in self.detached},
        }

    def summary(self) -> str:
        bits = []
        for name in ("added", "removed", "moved"):
            v = getattr(self, name)
            if v:
                bits.append(f"{name}: {', '.join(v)}")
        for name in ("attached", "detached"):
            v = getattr(self, name)
            if v:
                bits.append(f"{name}: " + "; ".join(" & ".join(p) for p in v))
        return " | ".join(bits) or "no tracked change"


class Step(StepOut):
    id: str
    diff_id: str
    start: FrameRef
    end: FrameRef


class ProcedureGraph(BaseModel):
    title: str
    steps: list[Step]
    preconditions: list[Precondition] = []
    any_order: list[list[str]] = []      # master-confirmed groups
    reviewed: bool = False


class LeakFinding(BaseModel):
    round: int
    frame: FrameRef
    item: LeakItem
    fix: Literal["box", "retrack", "none"]
    verified: Optional[bool] = None


class RedactionReceipt(BaseModel):
    concepts: dict[str, int]                 # concept -> number of tracked instances hidden
    frames_total: int
    frames_with_redaction: int
    leaks: list[LeakFinding] = []
    review_queue: list[FrameRef] = []
    excluded_obj_ids: list[str] = []          # manual click-to-unblur


class Mismatch(BaseModel):
    type: Literal["skipped", "order", "wrong_part", "extra"]
    step_id: Optional[str] = None
    detail: str
    master_frame: Optional[FrameRef] = None
    learner_frame: Optional[FrameRef] = None
    confirmed: Optional[bool] = None
    explanation: Optional[str] = None
    explanation_ur: Optional[str] = None


class Feedback(BaseModel):
    procedure_id: str
    matched: dict[str, str]                   # master step id -> learner diff id
    mismatches: list[Mismatch]
    passed: bool
    not_checked_tips: list[str] = []


class TraceEvent(BaseModel):
    stage: str
    kind: Literal["llm", "tool", "sam", "info", "fix"]
    message: str
    data: dict = {}
