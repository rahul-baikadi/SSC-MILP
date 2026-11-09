# Verifier run (Sonnet 5.5), one command:
#   powershell -ExecutionPolicy Bypass -File scripts\run_verify.ps1
# Reads the key from ..\anthropic_key.txt (or ANTHROPIC_API_KEY). Logs: logs\verify.log, logs\verify_status.txt
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$envDir = Join-Path $env:USERPROFILE "anaconda3\envs\ssc"
$py = Join-Path $envDir "python.exe"
$env:PATH = "$envDir;$envDir\Library\mingw-w64\bin;$envDir\Library\usr\bin;$envDir\Library\bin;$envDir\Scripts;$env:PATH"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = Join-Path $repo "logs\verify.log"
$status = Join-Path $repo "logs\verify_status.txt"
function Status($msg) { $line = "$(Get-Date -Format s)  $msg"; Add-Content $status $line; Write-Host "==> $msg" -ForegroundColor Cyan }
function Run($label, [string[]]$pyArgs) {
    Status "START $label"
    & $py @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE
    Status "END   $label (exit $code)"
    return $code
}
Set-Content $status "$(Get-Date -Format s)  verify run started"

& $py -c "import anthropic" 2>$null
if ($LASTEXITCODE -ne 0) { Status "installing anthropic SDK"; & $py -m pip install -q anthropic }

# 1. one-call authentication check (costs < 1 cent)
Status "START auth check"
$auth = & $py -c "import sys; sys.path.insert(0,'.'); from ssc.anthropic_client import AnthropicClient as A; c=A(); r=c._client.messages.create(model=c.model,max_tokens=5,messages=[{'role':'user','content':'Reply OK'}]); print('AUTH OK', r.model)" 2>&1 | ForEach-Object { "$_" }
$auth | Tee-Object -FilePath $log -Append | Out-Host
if (-not ($auth -match "AUTH OK")) { Status "FAIL: authentication (see logs\verify.log)"; exit 1 }
Status ("AUTH OK: " + (($auth | Select-String "AUTH OK") -join " "))

# 2. tests
if ((Run "tests" @("-m", "pytest", "tests\test_verify.py", "-q")) -ne 0) { Status "FAIL: tests"; exit 1 }

# 3. pilot: 10 MaMO-Complex problems (100 calls), then the rest
Run "pilot mamo_complex (10)" @("scripts\verify.py", "run", "--benchmark", "mamo_complex", "--n", "10") | Out-Null
Run "verify mamo_complex" @("scripts\verify.py", "run", "--benchmark", "mamo_complex") | Out-Null

# 4. OptMath: finish solving (skips solved problems), then verify
Run "solve optmath" @("scripts\verify.py", "solve", "--benchmark", "optmath", "--workers", "4") | Out-Null
Run "verify optmath" @("scripts\verify.py", "run", "--benchmark", "optmath") | Out-Null
Status "VERIFY DONE"
