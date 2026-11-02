"""Local hub web app. One GPU box serves every phone on the same Wi-Fi; nothing leaves the network.

Jobs run one at a time (single GPU) in a background worker thread.
"""
from __future__ import annotations

import json
import queue
import shutil
import socket
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT
from ..runctx import RunContext


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def create_app(cfg) -> FastAPI:
    app = FastAPI(title="USTAAD")
    runs = Path(cfg.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    uploads = runs / "_uploads"
    uploads.mkdir(exist_ok=True)
    jobs: dict[str, dict] = {}
    q: "queue.Queue[str]" = queue.Queue()

    def worker():
        from ..pipeline.learner import run_attempt
        from ..pipeline.master import run_master
        while True:
            jid = q.get()
            job = jobs[jid]
            job.update(state="running", started=time.time())

            def on_event(ev, job=job):
                job["stage"] = ev.stage
                job["events"].append(ev.model_dump())
                job["events"] = job["events"][-200:]
            try:
                if job["kind"] == "master":
                    run_master(job["video"], job["policy"], cfg, job["run_id"], on_event=on_event)
                else:
                    run_attempt(job["video"], job["procedure"], cfg, job["run_id"], on_event=on_event)
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001
                job.update(state="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:])
            job["finished"] = time.time()

    threading.Thread(target=worker, daemon=True).start()

    def _save_upload(f: UploadFile) -> Path:
        p = uploads / f"{uuid.uuid4().hex[:8]}_{Path(f.filename or 'video.mp4').name}"
        with p.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        return p

    def _job(kind: str, video: Path, **extra) -> dict:
        jid = uuid.uuid4().hex[:8]
        run_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{jid[:4]}"
        jobs[jid] = dict(id=jid, kind=kind, video=str(video), run_id=run_id, state="queued", stage=None,
                         events=[], queued=time.time(), **extra)
        q.put(jid)
        return {"job_id": jid, "run_id": run_id, "position": q.qsize()}

    @app.post("/api/master")
    def new_master(file: UploadFile = File(...), policy: str = Form("")):
        return _job("master", _save_upload(file), policy=policy)

    @app.post("/api/attempt/{procedure_id}")
    def new_attempt(procedure_id: str, file: UploadFile = File(...)):
        if not (runs / procedure_id / "procedure.json").exists():
            raise HTTPException(404, "procedure not found")
        return _job("attempt", _save_upload(file), procedure=procedure_id)

    @app.get("/api/jobs")
    def list_jobs():
        return [{k: v for k, v in j.items() if k not in ("events", "tb")} for j in jobs.values()]

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        if jid not in jobs:
            raise HTTPException(404)
        return jobs[jid]

    @app.get("/api/runs")
    def list_runs():
        out = []
        for d in sorted(runs.iterdir(), reverse=True):
            if d.name.startswith("_") or not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            title = None
            if (d / "procedure.json").exists():
                title = json.loads((d / "procedure.json").read_text()).get("title")
            out.append({"id": d.name, **meta, "status": status, "title": title})
        return out

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        d = runs / run_id
        if not d.exists():
            raise HTTPException(404)
        data = {}
        for name in ("meta", "status", "procedure", "receipt", "inventory", "feedback", "timings", "usage", "leak_audit"):
            p = d / f"{name}.json"
            if p.exists():
                data[name] = json.loads(p.read_text(encoding="utf-8"))
        data["trace"] = RunContext(cfg.runs_dir, run_id).trace()[-400:]
        if (d / "tracks.json").exists():
            objs: dict[str, dict] = {}
            refs = json.loads((d / "sample.json").read_text())
            for idx, lst in json.loads((d / "tracks.json").read_text()).items():
                for o in lst:
                    if o["role"] != "privacy":
                        continue
                    e = objs.setdefault(o["obj_id"], {"obj_id": o["obj_id"], "concept": o["concept"], "frames": 0,
                                                       "first_t": refs[int(idx)]["t"]})
                    e["frames"] += 1
                    e["first_t"] = min(e["first_t"], refs[int(idx)]["t"])
            data["privacy_objects"] = sorted(objs.values(), key=lambda o: o["first_t"])
        data["files"] = sorted(str(p.relative_to(d)) for p in d.glob("*.mp4")) + \
            sorted(str(p.relative_to(d)) for sub in ("card", "evidence") for p in (d / sub).glob("*.jpg"))
        return data

    @app.post("/api/runs/{run_id}/render")
    def rerender(run_id: str, exclude: list[str] = Body(default=[], embed=True)):
        """Click-to-unblur: re-render the privacy layer without the excluded object ids."""
        from ..pipeline.master import render_outputs
        rec = render_outputs(RunContext(cfg.runs_dir, run_id), cfg, set(exclude))
        return rec.model_dump(mode="json")

    @app.patch("/api/runs/{run_id}/procedure")
    def review(run_id: str, graph: dict = Body(...)):
        """Master review: save edited steps / any-order groups."""
        from ..schemas import ProcedureGraph
        g = ProcedureGraph.model_validate(graph)
        g.reviewed = True
        RunContext(cfg.runs_dir, run_id).save_json("procedure", g)
        return {"ok": True}

    @app.get("/api/lan")
    def lan(port: int = 7860):
        url = f"http://{lan_ip()}:{port}"
        svg = None
        try:
            import io

            import qrcode
            import qrcode.image.svg
            buf = io.BytesIO()
            qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage).save(buf)
            svg = buf.getvalue().decode()
        except Exception:
            pass
        return {"url": url, "qr_svg": svg}

    @app.get("/runs/{run_id}/{path:path}")
    def run_file(run_id: str, path: str, request: Request):
        p = (runs / run_id / path).resolve()
        if not str(p).startswith(str(runs.resolve())) or not p.is_file():
            raise HTTPException(404)
        if p.name == "source.mp4" and (runs / run_id / "meta.json").exists() and \
                json.loads((runs / run_id / "meta.json").read_text()).get("kind") == "master" and \
                request.client.host not in ("127.0.0.1", "::1", "localhost") and not cfg.get("serve_raw_master", False):
            # the raw master video is only viewable on the hub itself (for the privacy proof), never over Wi-Fi
            raise HTTPException(403, "raw master video is private; use redacted.mp4")
        return FileResponse(p)

    app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    return app
