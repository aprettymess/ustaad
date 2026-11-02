# Start Muse Glimmer on Windows by hand (llama.cpp CUDA build).
# Only needed if configs/local.yaml has glimmer.manage_server: false (otherwise USTAAD starts/stops it itself).
# RTX 5070 12 GB: Q4_K_M -> -ngl 28 (measured OK). UD-IQ3_XXS -> start 30, +2 steps (38 froze the PC).
# If raising -ngl FREEZES Windows: NVIDIA Control Panel > Manage 3D settings > Program Settings > llama-server.exe >
#   'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback'  (then it errors with OOM instead of freezing)
param(
  [string]$Model  = "F:\models\glimmer\Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
  [string]$MMProj = "F:\models\glimmer\mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
  [int]$NGL = 28,
  [int]$Ctx = 32768
)
if (-not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
  Write-Error "llama-server not found. winget install llama.cpp  (or unzip a CUDA 12.8+/13 release and add it to PATH)"; exit 1
}
llama-server -m $Model --mmproj $MMProj --no-mmproj-offload `
  -a muse-glimmer -ngl $NGL -c $Ctx -np 1 -fa on --jinja `
  --temp 1.0 --top-p 0.95 --top-k 64 `
  --host 0.0.0.0 --port 8080
