# Round-2 regeneration pilot, one command:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pilot.ps1
# Uses the 'ssc' conda env directly (no 'conda activate' needed), asks for the DeepSeek key at a
# hidden prompt (kept only in this process), and logs to logs\pilot.log + logs\pilot_status.txt.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$envDir = Join-Path $env:USERPROFILE "anaconda3\envs\ssc"
$py = Join-Path $envDir "python.exe"
$env:PATH = "$envDir;$envDir\Library\mingw-w64\bin;$envDir\Library\usr\bin;$envDir\Library\bin;$envDir\Scripts;$env:PATH"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = Join-Path $repo "logs\pilot.log"
$status = Join-Path $repo "logs\pilot_status.txt"
function Status($msg) { $line = "$(Get-Date -Format s)  $msg"; Add-Content $status $line; Write-Host "==> $msg" -ForegroundColor Cyan }
function Run($label, [string[]]$pyArgs) {
    Status "START $label"
    & $py @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE
    Status "END   $label (exit $code)"
    return $code
}

Set-Content $status "$(Get-Date -Format s)  pilot started"
if (-not (Test-Path $py)) { Status "FAIL: $py not found"; exit 1 }

# 1. environment
& $py -c "import pyomo, pyscipopt, openai, numpy, scipy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Status "installing missing packages into ssc"
    & (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe") install -n ssc -c conda-forge -y pyomo pyscipopt glpk numpy scipy openai pytest
}
if (-not (Get-Command glpsol -ErrorAction SilentlyContinue)) {
    & (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe") install -n ssc -c conda-forge -y glpk
}

# 2. key (hidden prompt, never written to disk)
if (-not $env:DEEPSEEK_API_KEY) {
    $s = Read-Host "Paste your DeepSeek key (input hidden), then press Enter" -AsSecureString
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))
    Remove-Variable s
}
if (-not $env:DEEPSEEK_API_KEY) { Status "FAIL: no key entered"; exit 1 }

# 3. tests + API check
if ((Run "tests" @("-m", "pytest", "tests", "-q")) -ne 0) { Status "FAIL: tests"; exit 1 }
Status "START check"
$chk = & $py scripts\run.py check 2>&1 | ForEach-Object { "$_" }
$chk | Tee-Object -FilePath $log -Append | Out-Host
if ($chk -match "FAILED") { Status "FAIL: run.py check (see logs\pilot.log)"; exit 1 }
Status ("CHECK OK: " + (($chk | Select-String "model=") -join " "))

# 4. re-score round 1 with current code
$benches = @("mamo_complex", "optmath")
foreach ($b in $benches) { Run "score-r1 $b" @("scripts\run.py", "score", "--benchmark", $b) | Out-Null }

# 5. pilot: three arms x two benchmarks
foreach ($b in $benches) {
    foreach ($a in @("blind", "feedback", "branch")) {
        Run "regen $b $a" @("scripts\regenerate.py", "run", "--benchmark", $b, "--arm", $a) | Out-Null
        Run "eval $b $a" @("scripts\run.py", "evaluate", "--benchmark", $b, "--run-name", "${b}_r2_$a") | Out-Null
        Run "merge $b $a" @("scripts\regenerate.py", "merge", "--benchmark", $b, "--arm", $a) | Out-Null
        Run "score-union $b $a" @("scripts\run.py", "score", "--benchmark", $b, "--run-name", "${b}_r2_${a}_union") | Out-Null
    }
}
Status "PILOT DONE"
