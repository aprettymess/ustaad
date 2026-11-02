# USTAAD

**Learn hands-on jobs from the people you trust. Privately.**

Someone who knows a household or vehicle job (your dad, a neighbour, a mechanic) films it once at home.
USTAAD turns the video into a step-by-step procedure, **automatically hides everything private**
(faces, licence plates, house numbers, documents, screens), and when a learner films their own attempt,
**shows exactly where it differs from the master**, with side-by-side evidence.

Everything runs **offline on one consumer GPU** with Meta's open-weight models. Nothing leaves the house.

> Built for the Meta Global AI Developer Hackathon (Cerebral Valley, Oct 26 – Nov 4, 2026) · Theme: *Build for Your World* · Track: Indie.
> Built in Pakistan, where the Meta Model API is geo-restricted, so it runs entirely on open weights:
> **Muse Glimmer 30B** (llama.cpp, quantized) + **SAM 3.1** (text-prompted video segmentation & tracking).

---

## Why

- Small jobs (AC filter service, a leaking tap, a flat tyre) cost money and waiting time, and the know-how
  disappears when the person who had it isn't around.
- Video tutorials are generic and one-way: they never tell *you* what you did wrong.
- People won't share videos filmed inside their homes. Family, documents and the street outside are all in shot.
  **Automatic, audited redaction is what makes home skill-sharing possible.**

## How it works

```
master video ─┬─ Glimmer: inventory (parts + privacy plan + hazard check)
              ├─ SAM 3.1: track task parts, hands and privacy concepts (one session, one prompt per concept)
              ├─ still moments → snapshots → deterministic state diffs        (SAM decides WHEN and WHAT changed)
              ├─ privacy render (dilated masks, temporal union, over-blur by default)
              ├─ Glimmer leak audit on REDACTED frames → box fix / SAM re-track → re-check  (agentic)
              ├─ Glimmer names each state change as a step                     (Glimmer decides WHAT it means)
              ├─ Glimmer verifier agent (tool calls: list / evidence / merge / delete / rename / finish)
              └─ Glimmer preconditions → procedure graph → master review → shareable card (redacted frames only)

learner video ─ SAM tracks → snapshots → diffs → match to master graph → order / skipped / wrong part
              └─ Glimmer judges each candidate mistake with paired frames → private feedback (EN + Urdu)
```

Design rules:
- **Claim only what a camera can verify.** Tightness, torque and feel become "master's tip — not checked".
- **Dependencies, not strict order.** Only true preconditions are flagged; the master marks "any order" groups.
- **Feedback, not grades.** No scores, no leaderboards, no reports to anyone else.
- **No biometrics.** No face recognition, no "don't blur me". Manual click-to-unblur only.
- **Hazard refusal.** Mains electricity, gas, medical procedures are refused at the inventory stage.
- **Raw master video never leaves the hub.** Other devices only ever get the redacted version.

## Quick start

### 0. No GPU? See the plumbing work in 30 seconds
```bash
pip install -r requirements.txt
python scripts/make_synthetic.py synthetic/
python -m ustaad.cli --mock master  synthetic/master.mp4 --policy "hide faces and plates" --run-id demo
python -m ustaad.cli --mock attempt synthetic/attempt_order.mp4 --procedure demo
python -m ustaad.cli --mock serve            # http://localhost:7860
pytest -q
```

### 1. Real models (native Windows 11, RTX 50-series)
See [`docs/setup.md`](docs/setup.md). Short version (PowerShell):
```powershell
py -3.12 -m venv F:\ustaad\venv ; F:\ustaad\venv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git ; pip install -e .\sam3     # after HF access to facebook/sam3.1
pip install -r requirements.txt ; pip install triton-windows                        # triton optional (shim built in)
copy configs\local.example.yaml configs\local.yaml                                # paths + tuned -ngl
python scripts\check_env.py --sam-video --glimmer                                  # everything PASS
powershell -File scripts\run_hub.ps1                                               # http://localhost:7860
```
Linux/WSL works too (`scripts/serve_glimmer.sh`).

## Repository map

| Path | What |
|---|---|
| `ustaad/pipeline/master.py` | master pipeline + GPU choreography (Glimmer ↔ SAM never co-resident on 12 GB) |
| `ustaad/pipeline/states.py` | still-moment snapshots and deterministic state diffs |
| `ustaad/pipeline/redact.py` | privacy render (full-fps video + sampled frames), click-to-unblur |
| `ustaad/pipeline/leak_audit.py` | agentic leak audit (detect → fix → re-check) |
| `ustaad/pipeline/procedure.py` | step naming, verifier agent, preconditions |
| `ustaad/pipeline/attempt.py`, `learner.py` | learner comparison and feedback |
| `ustaad/perception/sam.py`, `sam_compat.py` | SAM 3.1 wrapper (+ mock tracker) and native-Windows/no-Triton shim |
| `ustaad/glimmer/` | OpenAI-compatible client, tool-calling agent loop, prompts, mock, llama-server manager |
| `ustaad/server/app.py`, `web/` | local hub web app (phones on the same Wi-Fi via QR) |
| `scripts/check_env.py` | one-command readiness check (GPU, torch, SAM 3.1 video path, llama.cpp CUDA, Glimmer tool call) |
| `scripts/spikes.py` | day-0 go/no-go tests on real models |
| `scripts/bench_eval.py` | seeded-error and planted-PII benchmarks |
| `docs/` | setup, capture protocol, **task catalogue (15 home/bike/car jobs)**, architecture, privacy, demo script, results |

## Results

Filled from `scripts/bench_eval.py` on the published test set (`bench/`). Counts, not just percentages.

| Benchmark | Result |
|---|---|
| Mistake detection (seeded errors), precision / recall per type | _tbd_ |
| Redaction recall per concept (planted PII) | _tbd_ |
| Leaks caught by the auditor (before → after) | _tbd_ |
| Step accuracy before → after verifier | _tbd_ |
| Pipeline time per video minute (RTX 5070, Q4_K_M vs IQ3) | _tbd_ |

## Built during the hackathon

All code in this repository was written during the event (Oct 26 – Nov 4, 2026).
Third-party components are used as dependencies only: Muse Glimmer (Apache-2.0), SAM 3/3.1 (SAM License),
llama.cpp, FastAPI, OpenCV. No employer code is included.

## License

MIT for this repository's code. Model weights keep their own licences.
