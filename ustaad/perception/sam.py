"""SAM 3.1 text-prompted video tracking (+ a colour-blob mock for tests).

SAM 3 video API facts (facebookresearch/sam3):
- build_sam3_multiplex_video_predictor(...) is the SAM 3.1 (Object Multiplex) entry point.
  It defaults to use_fa3=True (FlashAttention-3, Hopper GPUs) -> we pass use_fa3=False for RTX 50-series.
- ONE text prompt per session at a time: reset_session -> add_prompt(text) -> propagate_in_video, per concept.
- resource_path can be a folder of <int>.jpg frames (what video.sample_frames writes).
- outputs per frame: out_obj_ids, out_probs, out_boxes_xywh (normalized), out_binary_masks (N,H,W bool).
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np

from ..schemas import TrackedObject
from .masks import MaskStore

Tracks = dict[int, list[TrackedObject]]  # sampled frame idx -> objects


class Sam31Tracker:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or (lambda *a, **k: None)
        self.predictor = None

    def _load(self):
        if self.predictor is None:
            from .sam_compat import apply
            mode = apply()  # Windows without Triton -> compatibility shim
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            self.log("sam", "sam", f"loading SAM 3.1 (multiplex), triton mode: {mode}")
            self.predictor = build_sam3_multiplex_video_predictor(
                max_num_objects=self.cfg.max_num_objects, use_fa3=self.cfg.use_fa3,
                compile=self.cfg.compile, default_output_prob_thresh=self.cfg.prob_thresh)

    def track(self, frames_dir: Path, concepts: Iterable[tuple[str, str]], store: MaskStore) -> Tracks:
        """concepts: (name, role). Returns per-frame objects; writes masks into store."""
        self._load()
        p = self.predictor
        sid = p.handle_request(dict(type="start_session", resource_path=str(frames_dir),
                                    offload_video_to_cpu=self.cfg.offload_video_to_cpu))["session_id"]
        tracks: Tracks = {}
        try:
            for name, role in concepts:
                p.handle_request(dict(type="reset_session", session_id=sid))
                p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=name))
                n_obj = set()
                for resp in p.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                    fi, out = resp["frame_index"], resp["outputs"]
                    masks = out["out_binary_masks"]
                    frame_masks = {}
                    for j, oid in enumerate(out["out_obj_ids"].tolist()):
                        key = f"{name}#{oid}"
                        m = masks[j].astype(bool)
                        n_obj.add(key)
                        frame_masks[key] = m
                        x, y, w, h = [float(v) for v in out["out_boxes_xywh"][j]]
                        tracks.setdefault(fi, []).append(TrackedObject(
                            obj_id=key, concept=name, role=role, box=[x, y, w, h],
                            area=float(m.mean()), prob=float(out["out_probs"][j])))
                    if frame_masks:
                        store.put(fi, frame_masks)
                self.log("sam", "sam", f"'{name}' -> {len(n_obj)} object(s)", concept=name, role=role)
        finally:
            p.handle_request(dict(type="close_session", session_id=sid))
        return tracks

    def unload(self):
        self.predictor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class MockTracker:
    """Finds solid-colour blobs in synthetic videos (scripts/make_synthetic.py). concept -> BGR colour."""
    COLORS = {"wheel": (200, 60, 30), "lug nut": (0, 220, 255), "jack": (40, 40, 220),
              "hand": (120, 170, 230), "face": (180, 120, 255), "licence plate": (245, 245, 245)}

    def __init__(self, cfg, log=None):
        self.log = log or (lambda *a, **k: None)

    def track(self, frames_dir: Path, concepts, store: MaskStore) -> Tracks:
        import cv2
        files = sorted(Path(frames_dir).glob("*.jpg"))
        tracks: Tracks = {}
        for f in files:
            fi = int(f.stem)
            img = cv2.imread(str(f)).astype(int)
            H, W = img.shape[:2]
            frame_masks = {}
            for name, role in concepts:
                col = self.COLORS.get(name)
                if col is None:
                    continue
                m = (np.abs(img - np.array(col)).sum(axis=2) < 60).astype(np.uint8)
                n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
                comps = sorted([i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 30],
                               key=lambda i: stats[i, cv2.CC_STAT_LEFT])
                for j, i in enumerate(comps):
                    key = f"{name}#{j}"
                    x, y, w, h = stats[i, :4]
                    frame_masks[key] = lab == i
                    tracks.setdefault(fi, []).append(TrackedObject(
                        obj_id=key, concept=name, role=role, box=[x / W, y / H, w / W, h / H],
                        area=float(stats[i, cv2.CC_STAT_AREA]) / (W * H)))
            if frame_masks:
                store.put(fi, frame_masks)
        for name, role in concepts:
            self.log("sam", "sam", f"'{name}' tracked (mock)", concept=name, role=role)
        return tracks

    def unload(self):
        pass


def get_tracker(cfg, log=None):
    return MockTracker(cfg, log) if cfg.backend == "mock" else Sam31Tracker(cfg, log)
