# Auto-resume watchdog for the phase-2 stage-1 full-pool run.
# Relaunches the runner if it is not running and the run is not yet complete
# (374 questions x 5 models = 1870 records). Safe to fire every few minutes.

$repo = "C:\Users\hello\llm-council-research\llm-council-governance"
$results = "$repo\experiments\results_phase2_stage1\stage1_results.json"
$log = "$repo\experiments\results_phase2_stage1\auto_resume.log"

function Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -Append -Encoding utf8 $log
}

# Already running? (match on the module name in the command line)
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "run_phase2_stage1" }
if ($running) { exit 0 }

# Complete? (count records cheaply; errored records count as done)
if (Test-Path $results) {
    try {
        $n = ([regex]::Matches((Get-Content $results -Raw -Encoding utf8), '"question_id"')).Count
    } catch { $n = 0 }
    if ($n -ge 1870) { Log "run complete ($n records) - nothing to do"; exit 0 }
} else {
    $n = 0
}

Log "runner not found at $n/1870 records - relaunching"
$env:PYTHONUTF8 = '1'
Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "experiments.run_phase2_stage1" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "$repo\experiments\results_phase2_stage1\run_stdout.log" `
    -RedirectStandardError "$repo\experiments\results_phase2_stage1\run_stderr.log" `
    -WindowStyle Hidden
