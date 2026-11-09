# Re-run only the cut-off verifier answers with a 4000-token cap:
#   powershell -ExecutionPolicy Bypass -File scripts\run_verify_redo.ps1
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$envDir = Join-Path $env:USERPROFILE "anaconda3\envs\ssc"
$py = Join-Path $envDir "python.exe"
$env:PATH = "$envDir;$envDir\Library\bin;$envDir\Scripts;$env:PATH"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = Join-Path $repo "logs\verify_redo.log"
$status = Join-Path $repo "logs\verify_status.txt"
function Status($msg) { $line = "$(Get-Date -Format s)  $msg"; Add-Content $status $line; Write-Host "==> $msg" -ForegroundColor Cyan }
foreach ($b in @("mamo_complex", "optmath")) {
    Status "START redo $b"
    & $py scripts\verify.py redo --benchmark $b --max-tokens 4000 --workers 8 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    Status "END   redo $b (exit $LASTEXITCODE)"
}
Status "REDO DONE"
