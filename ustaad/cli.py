"""USTAAD command line.

  python -m ustaad.cli master  VIDEO --policy "hide faces, plates and our house number"
  python -m ustaad.cli attempt VIDEO --procedure master-2026...
  python -m ustaad.cli render  RUN_ID --exclude "face#1"          # click-to-unblur from the CLI
  python -m ustaad.cli serve   [--host 0.0.0.0 --port 7860]       # web app (phones on the same Wi-Fi)

Add --mock to run without GPU models (synthetic tests, UI work).
"""
from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ustaad")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="mock Glimmer + SAM (no GPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("master")
    m.add_argument("video")
    m.add_argument("--policy", default="")
    m.add_argument("--run-id")
    m.add_argument("--force", action="store_true")
    a = sub.add_parser("attempt")
    a.add_argument("video")
    a.add_argument("--procedure", required=True)
    a.add_argument("--run-id")
    a.add_argument("--force", action="store_true")
    r = sub.add_parser("render")
    r.add_argument("run_id")
    r.add_argument("--exclude", nargs="*", default=[])
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    over = {"glimmer": {"backend": "mock"}, "sam": {"backend": "mock"}} if args.mock else {}
    cfg = load_config(args.config, **over)

    if args.cmd == "master":
        from .pipeline.master import run_master
        ctx = run_master(args.video, args.policy, cfg, args.run_id, args.force)
        print(json.dumps({"run_id": ctx.id, "dir": str(ctx.dir), "timings": ctx.timings}, indent=2))
    elif args.cmd == "attempt":
        from .pipeline.learner import run_attempt
        ctx = run_attempt(args.video, args.procedure, cfg, args.run_id, args.force)
        print(json.dumps({"run_id": ctx.id, "feedback": ctx.load("feedback")}, indent=2, ensure_ascii=False))
    elif args.cmd == "render":
        from .pipeline.master import render_outputs
        from .runctx import RunContext
        rec = render_outputs(RunContext(cfg.runs_dir, args.run_id), cfg, set(args.exclude))
        print(rec.model_dump_json(indent=2))
    elif args.cmd == "serve":
        import uvicorn
        from .server.app import create_app
        uvicorn.run(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
