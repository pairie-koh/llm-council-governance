# Auto-resume watchdog for the frontier full run.
# Registered as a scheduled task: relaunches the runner if it is not running
# and the run is not yet complete. Safe to fire every few minutes.

$repo = "C:\Users\hello\llm-council-research\llm-council-governance"
$results = "$repo\experiments\results_frontier_full\pilot_results.json"
$log = "$repo\experiments\results_frontier_full\auto_resume.log"

function Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -Append -Encoding utf8 $log
}

# Already running? (match on the module name in the command line)
$running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "run_frontier_full" }
if ($running) { exit 0 }

# Complete? (count records cheaply; errored records count as done)
if (Test-Path $results) {
    try {
        $n = ([regex]::Matches((Get-Content $results -Raw -Encoding utf8), '"question_id"')).Count
    } catch { $n = 0 }
    if ($n -ge 2210) { Log "run complete ($n trials) - nothing to do"; exit 0 }
} else {
    $n = 0
}

Log "runner not found at $n/2210 trials - relaunching"
$env:PYTHONUTF8 = '1'
Start-Process -FilePath "python" `
    -ArgumentList "-u", "-m", "experiments.run_frontier_full", "--skip-gpqa", "--max-concurrent", "12" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "$repo\experiments\results_frontier_full\run_stdout.log" `
    -RedirectStandardError "$repo\experiments\results_frontier_full\run_stderr.log" `
    -WindowStyle Hidden
