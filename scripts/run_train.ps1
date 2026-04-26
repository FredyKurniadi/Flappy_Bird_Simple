$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    throw "Virtual environment not found. Run .\scripts\setup_env.ps1 first."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& ".venv\Scripts\Activate.ps1"
python .\train_dqn_sb3.py
