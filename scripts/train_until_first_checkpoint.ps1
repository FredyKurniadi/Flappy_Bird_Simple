$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    throw "Virtual environment not found. Run .\scripts\setup_env.ps1 first."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& ".venv\Scripts\Activate.ps1"

$env:TOTAL_TIMESTEPS = if ($env:TOTAL_TIMESTEPS) { $env:TOTAL_TIMESTEPS } else { "50000" }
$env:CHECKPOINT_FREQ = if ($env:CHECKPOINT_FREQ) { $env:CHECKPOINT_FREQ } else { "50000" }
$env:EVAL_FREQ = if ($env:EVAL_FREQ) { $env:EVAL_FREQ } else { "10000" }
$env:N_EVAL_EPISODES = if ($env:N_EVAL_EPISODES) { $env:N_EVAL_EPISODES } else { "5" }

python .\train_dqn_sb3.py

$checkpointDir = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { "artifacts/checkpoints" }
$firstCheckpoint = Get-ChildItem -Path $checkpointDir -Filter "dqn_ckpt_*.zip" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -First 1

if ($null -eq $firstCheckpoint) {
    throw "Training finished but no checkpoint found in $checkpointDir"
}

Write-Host "First checkpoint found: $($firstCheckpoint.FullName)"
