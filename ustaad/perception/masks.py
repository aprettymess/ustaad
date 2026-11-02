"""Per-frame mask storage on disk (bit-packed) so long videos don't blow RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MaskStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, idx: int) -> Path:
        return self.root / f"{idx:05d}.npz"

    def put(self, idx: int, masks: dict[str, np.ndarray]) -> None:
        """Merge masks for frame idx (obj_id -> bool HxW)."""
        cur = self.get(idx)
        cur.update(masks)
        if not cur:
            return
        shape = next(iter(cur.values())).shape
        np.savez_compressed(self._p(idx), __shape=np.array(shape),
                            **{k: np.packbits(v.astype(bool)) for k, v in cur.items()})

    def get(self, idx: int) -> dict[str, np.ndarray]:
        p = self._p(idx)
        if not p.exists():
            return {}
        z = np.load(p)
        shape = tuple(z["__shape"])
        n = int(np.prod(shape))
        return {k: np.unpackbits(z[k])[:n].reshape(shape).astype(bool) for k in z.files if k != "__shape"}

    def union(self, idx: int, keys: set[str] | None = None, exclude: set[str] | None = None,
              shape: tuple[int, int] | None = None) -> np.ndarray | None:
        ms = self.get(idx)
        sel = [m for k, m in ms.items() if (keys is None or k in keys) and not (exclude and k in exclude)]
        if not sel:
            return np.zeros(shape, bool) if shape else None
        out = np.zeros_like(sel[0], dtype=bool)
        for m in sel:
            out |= m
        return out
