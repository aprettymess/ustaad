# Architecture notes

## Why still moments + state diffs
Hands occlude parts and motion is noisy. Instead of recognising actions from motion, USTAAD finds moments where the
scene is still (low frame difference, hand masks ≈ 0), snapshots tracked objects there, and diffs consecutive
snapshots: added / removed / moved / attached / detached per concept. The diff is deterministic and cheap.
Glimmer is only asked to *name* a measured change from a BEFORE/AFTER pair. That's a much easier and more
checkable task than open-ended action recognition.

## Run directory (`runs/<id>/`)
`source.mp4` (normalised H.264) · `frames/` sampled JPEGs (SAM input) · `masks/` bit-packed per-frame masks ·
`sample.json` `inventory.json` `tracks.json` `moments.json` `diffs.json` `leak_audit.json` `fixes.json`
`receipt.json` `steps_draft.json` `steps.json` `procedure.json` · `redacted/` `redacted.mp4` `overlay.mp4` `card/` ·
`trace.jsonl` (glass box) · `timings.json` `usage.json`. Every stage is cached; `--force` reruns.

## SAM 3.1 facts baked into the wrapper
- `build_sam3_multiplex_video_predictor(use_fa3=False, ...)`: the SAM 3.1 entry point. FA3 is Hopper-only.
- One text prompt per session at a time: reset → add_prompt(text) → propagate, per concept (cost scales with concepts).
- Outputs per frame: `out_obj_ids`, `out_probs`, `out_boxes_xywh` (normalised), `out_binary_masks`.

## Glimmer facts baked into the client
- `chat_template_kwargs.reasoning_strength` (llama.cpp ignores `reasoning_effort`). Reasoning counts against `max_tokens`.
- Structured outputs via `response_format: json_schema` + one repair retry.
- Tool results must be text (llama.cpp rejects images in tool results), so images go in user messages.
- Never send a custom `stop` list (stopping on `<|eom|>` breaks tool calling).
