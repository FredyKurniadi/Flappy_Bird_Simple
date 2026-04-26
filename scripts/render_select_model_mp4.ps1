param(
    [ValidateSet("best", "final", "checkpoint", "custom")]
    [string]$ModelSource = "best",
    [string]$ModelPath = "",
    [int]$CheckpointSteps = 3000000,
    [int]$Episodes = 3,
    [string]$OutputName = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$localVenvPython = ".venv\Scripts\python.exe"
$parentVenvPython = "..\.venv\Scripts\python.exe"

if (Test-Path $localVenvPython) {
    $pythonExe = $localVenvPython
} elseif (Test-Path $parentVenvPython) {
    $pythonExe = $parentVenvPython
} else {
    throw "Python executable from venv not found. Run .\scripts\setup_env.ps1 first."
}

$modelDir = "artifacts/models"
$checkpointDir = "artifacts/checkpoints"

switch ($ModelSource) {
    "best" {
        $resolvedModelPath = Join-Path $modelDir "best_model.zip"
    }
    "final" {
        $modelName = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "dqn_flappy_model" }
        $resolvedModelPath = Join-Path $modelDir "$modelName.zip"
    }
    "checkpoint" {
        $resolvedModelPath = Join-Path $checkpointDir "dqn_ckpt_${CheckpointSteps}_steps.zip"
    }
    "custom" {
        if ([string]::IsNullOrWhiteSpace($ModelPath)) {
            throw "ModelSource custom requires -ModelPath."
        }
        $resolvedModelPath = $ModelPath
    }
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "Selected model file not found: $resolvedModelPath"
}

if ([string]::IsNullOrWhiteSpace($OutputName)) {
    $safeSource = $ModelSource
    if ($ModelSource -eq "checkpoint") {
        $safeSource = "ckpt_${CheckpointSteps}"
    }
    $OutputName = "flappy_${safeSource}_${Episodes}ep.mp4"
}

$env:MODEL_PATH = $resolvedModelPath
$env:RENDER_EPISODES = "$Episodes"
$env:OUTPUT_VIDEO_NAME = $OutputName

& $pythonExe ".\render_multi_episode.py"

$outputPath = Join-Path "artifacts/videos" $OutputName
if (-not (Test-Path $outputPath)) {
    throw "Render finished but output video not found: $outputPath"
}

Get-Item $outputPath | Select-Object Name, Length, LastWriteTime | Format-List
