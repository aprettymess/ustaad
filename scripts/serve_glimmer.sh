#!/usr/bin/env bash
# Start Muse Glimmer inside WSL/Linux (llama.cpp). Edit paths / NGL for your GPU.
set -euo pipefail
MODEL="${MODEL:-$HOME/models/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/glimmer/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf}"
NGL="${NGL:-28}"
BIN="$(command -v llama-server || command -v llama || true)"
[ -z "$BIN" ] && { echo "install llama.cpp: curl -LsSf https://llama.app/install.sh | sh"; exit 1; }
SUB=""; [ "$(basename "$BIN")" = "llama" ] && SUB="serve"
exec "$BIN" $SUB -m "$MODEL" --mmproj "$MMPROJ" --no-mmproj-offload \
  -a muse-glimmer -ngl "$NGL" -c 32768 -np 1 -fa on --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 --host 0.0.0.0 --port 8080
