param(
    [string]$PythonExe = "",
    [string]$VenvDir = ".venv",
    [string]$Requirements = "requirements.txt"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    try {
        $pyVersion = & py -3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $pyVersion -match "Python 3\.11") {
            $PythonExe = "py -3.11"
        }
    } catch {
        # Fallback handled below.
    }

    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $pythonVersion -match "Python 3\.11") {
            $PythonExe = "python"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    throw "Python 3.11 executable not found. Install Python 3.11 or pass -PythonExe explicitly."
}

$pythonVersion = Invoke-Expression "$PythonExe --version"
if ($pythonVersion -notmatch "Python 3\.11") {
    throw "Python 3.11 is required. Current: $pythonVersion"
}

Write-Host "[1/4] Creating virtual environment: $VenvDir"
Invoke-Expression "$PythonExe -m venv $VenvDir"

Write-Host "[2/4] Activating environment"
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "[3/4] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

Write-Host "[4/4] Installing requirements from $Requirements"
pip install -r $Requirements

Write-Host "Environment setup completed."
Write-Host "Next: .\scripts\export_versions.ps1"
