"""A run = one processed video. Holds paths, stage cache and the glass-box trace."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, TypeAdapter

from .schemas import TraceEvent

T = TypeVar("T")


class RunContext:
    def __init__(self, runs_dir: str, run_id: Optional[str] = None, kind: str = "master",
                 force: bool = False, on_event: Optional[Callable[[TraceEvent], None]] = None):
        self.id = run_id or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.kind = kind
        self.dir = Path(runs_dir) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.on_event = on_event
        self.trace_path = self.dir / "trace.jsonl"
        self.timings: dict[str, float] = self._load("timings.json", dict) or {}

    # ---- paths
    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def frames_dir(self) -> Path:
        d = self.dir / "frames"
        d.mkdir(exist_ok=True)
        return d

    # ---- trace (glass box)
    def log(self, stage: str, kind: str, message: str, **data: Any) -> None:
        ev = TraceEvent(stage=stage, kind=kind, message=message, data=_jsonable(data))
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(ev.model_dump_json() + "\n")
        print(f"[{stage}:{kind}] {message}")
        if self.on_event:
            self.on_event(ev)

    def trace(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        return [json.loads(l) for l in self.trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---- stage cache
    def stage(self, name: str, fn: Callable[[], T], out_type: Any = None) -> T:
        """Run fn once per run; cache its (pydantic/json) result in <run>/<name>.json."""
        p = self.dir / f"{name}.json"
        if p.exists() and not self.force:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return TypeAdapter(out_type).validate_python(raw) if out_type is not None else raw
        t0 = time.time()
        self.log(name, "info", "start")
        result = fn()
        self.save_json(name, result)
        self.timings[name] = round(time.time() - t0, 2)
        self.save_json("timings", self.timings)
        self.log(name, "info", f"done in {self.timings[name]}s")
        return result

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / f"{name}.json"
        p.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def _load(self, name: str, default_type: Type = dict) -> Any:
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def load(self, name: str, out_type: Any = None) -> Any:
        raw = json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
        return TypeAdapter(out_type).validate_python(raw) if out_type is not None else raw


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj
