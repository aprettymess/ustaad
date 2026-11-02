# Day-0 spike results (fill in as you run `scripts/spikes.py`)

| Spike | Command | Pass bar | Result | Decision |
|---|---|---|---|---|
| SAM tracking | `spikes.py sam clip.mp4 --concepts ...` | all concepts seen; ids ≈ real count | | |
| Glimmer state naming | `spikes.py diff ...` ×4 | ≥ 3/4 correct | | |
| Leak audit | `spikes.py leak frame.jpg` | planted leak boxed | | |
| Urdu | `spikes.py urdu` | ≥ 4/5 clear | | bilingual / English-only |
| Pipeline time | `spikes.py time clip.mp4` | < 10 min for a 2-min clip | | Q4_K_M / IQ3 |

Fallbacks: SAM loses ids → per-snapshot matching by label + IoU. Glimmer misnames → template titles from diff types,
Glimmer only for wording. Too slow → IQ3 quant, `sampling.fps: 2`, `reasoning_strength: low`.
