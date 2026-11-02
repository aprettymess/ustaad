# Setup: native Windows 11, RTX 5070 (12 GB)

Everything runs natively on Windows: llama.cpp (Muse Glimmer), SAM 3.1, and the USTAAD hub.
SAM 3 was confirmed working on native Windows (RTX 4060 work PC). Two Windows-specific details are handled by USTAAD:
- **Triton:** SAM's video tracker imports Triton at load time. Install `triton-windows`, or let USTAAD's shim
  (`ustaad/perception/sam_compat.py`) stub the one Triton-only function (interactive point sampling, unused by us)
  and use CPU NMS.
- **FlashAttention-3:** `sam.use_fa3: false` (already the default). FA3 targets Hopper GPUs.

Run `python scripts/check_env.py` after each step. The goal is all PASS.

## Paths used
| What | Where |
|---|---|
| Python venv | `F:\ustaad\venv` |
| Repo | `F:\ustaad` |
| Glimmer GGUFs | `F:\models\glimmer\` |
| Hugging Face cache (SAM weights) | `F:\hf-cache` (`HF_HOME`) |
| Footage | `F:\ustaad\data\` |

## Steps
1. NVIDIA driver ≥ 580 (`nvidia-smi` shows CUDA ≥ 12.8).
2. `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg llama.cpp OBSProject.OBSStudio`
3. Long paths + HF cache + Defender exclusions (PowerShell **Admin**):
   ```powershell
   New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
   setx HF_HOME F:\hf-cache
   Add-MpPreference -ExclusionPath F:\models, F:\hf-cache, F:\ustaad
   ```
4. venv + torch (cu128 = Blackwell-ready):
   ```powershell
   py -3.12 -m venv F:\ustaad\venv ; F:\ustaad\venv\Scripts\Activate.ps1
   python -m pip install -U pip
   pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
5. SAM 3 + Triton:
   ```powershell
   cd F:\ustaad ; git clone https://github.com/facebookresearch/sam3.git ; pip install -e .\sam3
   pip install -U "huggingface_hub[cli]" ; hf auth login
   pip install triton-windows        # optional; pick the version matching your torch (see triton-windows README)
   ```
6. Glimmer weights:
   ```powershell
   hf download meta-models/Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf mmproj-Muse-Glimmer-30B-Q4_K_M.gguf --local-dir F:\models\glimmer
   hf download unsloth/Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-UD-IQ3_XXS.gguf --local-dir F:\models\glimmer
   ```
7. USTAAD:
   ```powershell
   cd F:\ustaad ; git clone https://github.com/aprettymess/ustaad.git ; cd ustaad
   pip install -r requirements.txt
   copy configs\local.example.yaml configs\local.yaml    # edit -ngl and paths
   pytest
   python scripts\check_env.py --sam-video --glimmer
   ```
8. Hub: `powershell -File scripts\run_hub.ps1` → http://localhost:7860 → **Phone link** for the QR code.
   Windows Firewall: allow python.exe and llama-server.exe on **Private** networks only.

## GPU layers (`-ngl`) on 12 GB
Measured: Q4_K_M runs well at `-ngl 28`. IQ3_XXS at 38 froze the PC, so start it at 30 and go up by 2.
To make a too-high value fail fast instead of freezing Windows, open NVIDIA Control Panel → Manage 3D settings →
Program Settings → `llama-server.exe` → **CUDA – Sysmem Fallback Policy = Prefer No Sysmem Fallback**.

## Fitting Glimmer + SAM in 12 GB
`glimmer.manage_server: true` (default in `local.example.yaml`): USTAAD stops llama-server before SAM stages and
restarts it after. The pipeline is ordered to minimise swaps: inventory (G) → track (S) → audit (G) [→ re-track (S) → G] → steps (G).
If you start llama-server by hand (`scripts/serve_glimmer.ps1`), set `manage_server: false`, and lower `-ngl`
or stop it yourself while SAM runs.
