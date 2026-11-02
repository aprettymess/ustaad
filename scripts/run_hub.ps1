# Start the USTAAD hub on Windows. Phones on the same Wi-Fi open the "Phone link" QR.
# First run: Windows Firewall asks about python.exe -> allow on PRIVATE networks only.
param([int]$Port = 7860, [string]$Venv = "F:\ustaad\venv")
& "$Venv\Scripts\Activate.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m ustaad.cli serve --host 0.0.0.0 --port $Port
