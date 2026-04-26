param(
    [string]$OutFile = "VERSIONS_RUNTIME.md"
)

$ErrorActionPreference = "Stop"

$pythonVersion = python --version 2>&1
$pipVersion = pip --version 2>&1
$freeze = pip freeze --all

@"
# Runtime Versions (Generated)

Generated at: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Core Tooling
- $pythonVersion
- $pipVersion

## Full Installed Packages

```text
$freeze
```
"@ | Set-Content -Path $OutFile -Encoding UTF8

Write-Host "Saved runtime versions to $OutFile"
