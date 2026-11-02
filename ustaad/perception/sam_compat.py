"""Make SAM 3.1's video stack import and run on native Windows without Triton.

Why this exists (checked in facebookresearch/sam3):
- sam3/model/sam3_tracker_utils.py does `from sam3.model.edt import edt_triton` at import time, and edt.py
  does `import triton`. Official Triton has no Windows wheels. edt_triton is only used for *interactive point
  sampling* (sample_one_point_from_error_center) - never by text-prompted tracking, which is all USTAAD uses.
- sam3.perflib ("USE_PERFLIB", on by default) can route mask NMS / connected components to Triton kernels on CUDA.
- The SAM 3.1 multiplex builder uses fill_hole_area=0, so the connected-components path is skipped at runtime.

Strategy:
  1. If `import triton` works (Linux, or `pip install triton-windows` on Windows) -> do nothing.
  2. Otherwise: USE_PERFLIB=0, stub sam3.model.edt, and route sam3.perflib.nms.generic_nms to its CPU version.
Must be called BEFORE anything from `sam3` is imported.
"""
from __future__ import annotations

import os
import sys
import types

_applied: str | None = None


def apply() -> str:
    global _applied
    if _applied:
        return _applied
    try:
        import triton  # noqa: F401
        _applied = "triton"
        return _applied
    except Exception:
        pass

    os.environ.setdefault("USE_PERFLIB", "0")

    edt = types.ModuleType("sam3.model.edt")

    def edt_triton(data):  # pragma: no cover - only hit by interactive point prompts
        raise NotImplementedError("edt_triton needs Triton (interactive point sampling); "
                                  "install `triton-windows` if you need point prompts.")

    edt.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = edt

    try:
        import sam3.perflib.nms as nms
        if not getattr(nms, "GENERIC_NMS_AVAILABLE", False):
            nms.generic_nms = lambda ious, scores, iou_threshold=0.5: nms.generic_nms_cpu(ious, scores, iou_threshold)
    except Exception:
        pass
    _applied = "shim"
    return _applied
