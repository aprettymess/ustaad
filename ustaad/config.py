"""Config loading: configs/default.yaml <- configs/local.yaml <- USTAAD__SECTION__KEY env vars."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(val: str) -> Any:
    try:
        return yaml.safe_load(val)
    except Exception:
        return val


class Config(dict):
    """dict with attribute access: cfg.glimmer.base_url"""

    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v


def load_config(path: str | os.PathLike | None = None, **overrides) -> Config:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    local = ROOT / "configs" / "local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    if path:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for key, val in os.environ.items():
        if key.startswith("USTAAD__"):
            parts = key[len("USTAAD__"):].lower().split("__")
            node = cfg
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _coerce(val)
    cfg = _deep_merge(cfg, overrides)
    runs = Path(cfg["runs_dir"])
    cfg["runs_dir"] = str(runs if runs.is_absolute() else ROOT / runs)
    return Config(cfg)
