"""Start/stop llama-server around SAM stages so both fit in 12 GB (glimmer.manage_server: true).

Works on Windows and Linux. `glimmer.server_cmd` may be a list (recommended on Windows: no quoting issues)
or a single string.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.request
from contextlib import contextmanager

_proc: subprocess.Popen | None = None
IS_WIN = os.name == "nt"


def _healthy(base_url: str) -> bool:
    root = base_url.rsplit("/v1", 1)[0]
    try:
        req = urllib.request.Request(root + "/health")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy
        with opener.open(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _argv(cmd) -> list[str]:
    if isinstance(cmd, (list, tuple)):
        return [str(c) for c in cmd]
    return shlex.split(cmd, posix=not IS_WIN)


def ensure_up(cfg, log=None) -> None:
    global _proc
    if not cfg.manage_server or _healthy(cfg.base_url):
        return
    if not cfg.server_cmd:
        raise RuntimeError("glimmer.manage_server is true but glimmer.server_cmd is empty")
    log and log("glimmer", "info", "starting llama-server")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if IS_WIN else 0
    logf = open(os.path.join(os.path.expanduser("~"), "ustaad-llama-server.log"), "ab")
    _proc = subprocess.Popen(_argv(cfg.server_cmd), stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < cfg.server_ready_timeout_s:
        if _healthy(cfg.base_url):
            log and log("glimmer", "info", f"llama-server ready in {time.time() - t0:.0f}s")
            return
        if _proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup - see ~/ustaad-llama-server.log")
        time.sleep(2)
    raise TimeoutError("llama-server did not become healthy (see ~/ustaad-llama-server.log)")


def ensure_down(cfg, log=None) -> None:
    """Stop the server we started. If the user started it by hand, we leave it alone (and warn)."""
    global _proc
    if not cfg.manage_server:
        return
    if _proc is None:
        if _healthy(cfg.base_url):
            log and log("glimmer", "info", "llama-server was started by hand; stop it before SAM if VRAM runs out")
        return
    log and log("glimmer", "info", "stopping llama-server to free VRAM for SAM")
    _proc.terminate()  # TerminateProcess on Windows, SIGTERM on Linux
    try:
        _proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _proc.kill()
    _proc = None
    time.sleep(2)  # let the driver release VRAM


@contextmanager
def glimmer_on(cfg, log=None):
    ensure_up(cfg, log)
    yield


@contextmanager
def glimmer_off(cfg, log=None):
    ensure_down(cfg, log)
    yield
