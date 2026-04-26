$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    .\scripts\setup_env.ps1
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

.\scripts\train_until_first_checkpoint.ps1
.\scripts\render_best_mp4.ps1
.\scripts\export_versions.ps1

Write-Host "Pipeline complete: checkpoint, best/final model, metrics, and mp4 video are ready."
