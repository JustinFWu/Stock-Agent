# Scheduled-task entry point: run the daily session, then put the PC back to sleep.
#
# The task wakes the machine (WakeToRun), but nothing returns it to sleep afterwards, and the
# AC power plan never sleeps on idle. Sleep only when the machine was asleep just before this
# run started, so a scheduled run never suspends a PC someone was already using.

param([string[]]$DailyArgs = @('--submit', '--strategy', 'equal'))

$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo 'data\state\daily\sleep.log'
$started = Get-Date

function Write-Log([string]$message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $message" | Add-Content -Path $log -Encoding utf8
}

# The most recent resume. Event 1 from Power-Troubleshooter is logged on every wake from sleep.
$wake = Get-WinEvent -FilterHashtable @{
    LogName = 'System'; ProviderName = 'Microsoft-Windows-Power-Troubleshooter'; Id = 1
} -MaxEvents 1 -ErrorAction SilentlyContinue
$wokeForThisRun = $wake -and ($started - $wake.TimeCreated).TotalMinutes -le 5

Set-Location $repo
& (Join-Path $repo 'venv\Scripts\stock-agent-daily.exe') @DailyArgs
$code = $LASTEXITCODE
Write-Log "stock-agent-daily exited $code"

if ($wokeForThisRun) {
    Write-Log "woke at $($wake.TimeCreated); suspending"
    Add-Type -AssemblyName System.Windows.Forms
    # Suspend (not hibernate), not forced, wake timers stay enabled for the next run.
    [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false) | Out-Null
} else {
    Write-Log "machine was already awake; leaving it on"
}

exit $code
