"""One-command readiness check. Run it after each setup step; everything should end up PASS.

  python scripts/check_env.py                 # fast checks
  python scripts/check_env.py --sam-video     # + builds SAM 3.1 video predictor and tracks 8 frames (the real path)
  python scripts/check_env.py --glimmer       # + chat + tool-call round trip against llama-server
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
rows: list[tuple[str, str, str]] = []


def row(name, ok, detail=""):
    status = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
    rows.append((status, name, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-video", action="store_true")
    ap.add_argument("--glimmer", action="store_true")
    a = ap.parse_args()

    row("python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0])

    from ustaad.config import load_config
    cfg = load_config()
    row("config loads", True, f"runs_dir={cfg.runs_dir}")
    free = shutil.disk_usage(Path(cfg.runs_dir).anchor or "/").free / 2**30
    row("free disk (runs drive)", free > 30 or None, f"{free:.0f} GB free")
    row("ffmpeg on PATH", bool(shutil.which("ffmpeg")) or None,
        shutil.which("ffmpeg") or "missing: winget install Gyan.FFmpeg (needed for phone videos + browser playback)")

    try:
        import torch
        ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, CUDA {torch.version.cuda}"
        if ok:
            cap = torch.cuda.get_device_capability(0)
            detail += f", {torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
            if cap >= (12, 0) and tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8):
                ok, detail = False, detail + " -> Blackwell needs a cu128+ torch build"
        row("torch + CUDA GPU", ok, detail)
    except Exception as e:
        row("torch + CUDA GPU", False, str(e))

    try:
        import triton
        row("triton", True, f"triton {triton.__version__}")
    except Exception:
        row("triton", None, "not installed -> USTAAD uses its Windows shim (fine for text-prompted tracking); "
                           "optional: pip install triton-windows")

    try:
        from ustaad.perception.sam_compat import apply
        mode = apply()
        import sam3  # noqa: F401
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: F401
        row("sam3 importable (video stack)", True, f"mode={mode}")
    except Exception as e:
        row("sam3 importable (video stack)", False, f"{type(e).__name__}: {e}")

    try:
        from huggingface_hub import try_to_load_from_cache
        for repo, f in (("facebook/sam3", "sam3.pt"), ("facebook/sam3.1", "sam3.1_multiplex.pt")):
            p = try_to_load_from_cache(repo, f)
            row(f"checkpoint {repo}", isinstance(p, str) or None,
                p if isinstance(p, str) else "not cached yet (downloads on first use; needs HF access + hf auth login)")
    except Exception as e:
        row("checkpoints", None, str(e))

    llama = shutil.which("llama-server") or shutil.which("llama")
    sc = cfg.glimmer.server_cmd
    if not llama and isinstance(sc, list) and sc:
        llama = shutil.which(sc[0]) or (sc[0] if Path(sc[0]).exists() else None)
    row("llama-server found", bool(llama) or None, llama or "not on PATH (winget install llama.cpp)")
    if llama:
        try:
            out = subprocess.run([llama, "--list-devices"], capture_output=True, text=True, timeout=60)
            txt = (out.stdout + out.stderr)
            row("llama.cpp sees CUDA", "CUDA" in txt or None, " ".join(l.strip() for l in txt.splitlines() if "CUDA" in l)[:160]
                or "no CUDA device listed -> use a CUDA 12.8+/13 Windows release build")
        except Exception as e:
            row("llama.cpp sees CUDA", None, str(e))

    from ustaad.glimmer.server import _healthy
    up = _healthy(cfg.glimmer.base_url)
    row("Glimmer server healthy", up or None, cfg.glimmer.base_url + ("" if up else " (not running right now)"))

    if a.glimmer and up:
        try:
            from openai import OpenAI
            c = OpenAI(base_url=cfg.glimmer.base_url, api_key=cfg.glimmer.api_key)
            t0 = time.time()
            r = c.chat.completions.create(model=cfg.glimmer.model, max_tokens=2048, messages=[
                {"role": "user", "content": "What is the weather in Lahore in celsius? Use the tool."}],
                tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                extra_body={"chat_template_kwargs": {"reasoning_strength": "low"}})
            tc = r.choices[0].message.tool_calls
            tps = (r.usage.completion_tokens / (time.time() - t0)) if r.usage else 0
            row("Glimmer tool call", bool(tc), (f"{tc[0].function.name}({tc[0].function.arguments})" if tc else "no tool_calls")
                + f", ~{tps:.1f} tok/s")
        except Exception as e:
            row("Glimmer tool call", False, str(e))

    if a.sam_video:
        try:
            import cv2
            import torch
            from ustaad.perception.masks import MaskStore
            from ustaad.perception.sam import Sam31Tracker
            from ustaad.video import sample_frames
            tmp = Path(tempfile.mkdtemp(prefix="ustaad_check_"))
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), str(tmp)], check=True,
                           capture_output=True)
            refs = sample_frames(tmp / "master.mp4", tmp / "frames", 1, 640)[:8]
            for p in (tmp / "frames").glob("*.jpg"):
                if int(p.stem) >= 8:
                    p.unlink()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tr = Sam31Tracker(cfg.sam)
            tracks = tr.track(tmp / "frames", [("blue square", "task"), ("red rectangle", "task")], MaskStore(tmp / "masks"))
            n = len({o.obj_id for objs in tracks.values() for o in objs})
            peak = torch.cuda.max_memory_allocated() / 2**30
            tr.unload()
            row("SAM 3.1 video tracking (8 frames)", True,
                f"{n} object track(s), {time.time() - t0:.0f}s incl. load, peak VRAM {peak:.1f} GB")
        except Exception as e:
            row("SAM 3.1 video tracking (8 frames)", False, f"{type(e).__name__}: {e}")

    w = max(len(r[1]) for r in rows)
    print()
    for s, n, d in rows:
        print(f"[{s}] {n:<{w}}  {d}")
    fails = sum(r[0] == "FAIL" for r in rows)
    print(f"\n{fails} FAIL, {sum(r[0] == 'WARN' for r in rows)} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
