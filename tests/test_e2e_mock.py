"""Full master -> attempt run on synthetic videos with mock Glimmer + mock SAM (no GPU)."""
import subprocess
import sys
from pathlib import Path

from ustaad.config import load_config
from ustaad.pipeline.learner import run_attempt
from ustaad.pipeline.master import run_master

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), str(tmp_path)], check=True)
    cfg = load_config(runs_dir=str(tmp_path / "runs"), glimmer={"backend": "mock"}, sam={"backend": "mock"})
    m = run_master(tmp_path / "master.mp4", "hide faces and plates", cfg, run_id="m")
    g = m.load("procedure")
    assert [s["title"] for s in g["steps"]] == ["Loosen and remove the lug nuts", "Place the jack under the car",
                                               "Remove the wheel"]
    assert m.load("receipt")["concepts"] == {"face": 1, "licence plate": 1}
    assert (m.dir / "redacted.mp4").exists() and (m.dir / "card" / "s1_after.jpg").exists()

    bad = run_attempt(tmp_path / "attempt_order.mp4", "m", cfg, run_id="a_bad").load("feedback")
    assert not bad["passed"] and [x["type"] for x in bad["mismatches"]] == ["order"]
    ok = run_attempt(tmp_path / "attempt_ok.mp4", "m", cfg, run_id="a_ok").load("feedback")
    assert ok["passed"]
