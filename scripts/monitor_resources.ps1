[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RunLabel,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$IntervalMilliseconds = 500,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$DurationSeconds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-CsvField {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    return '"' + $Value.Replace('"', '""') + '"'
}

function Format-InvariantNumber {
    param(
        [double]$Value
    )

    return $Value.ToString(
        "0.000",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Format-InvariantInteger {
    param(
        [int]$Value
    )

    return $Value.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Get-ProcessAggregate {
    param(
        [System.Diagnostics.Process[]]$Processes,

        [Parameter(Mandatory = $true)]
        [hashtable]$LastCpuByProcess,

        [double]$PreviousCpuTotal
    )

    [int64]$workingSetBytes = 0
    [int64]$privateMemoryBytes = 0
    [int]$processCount = 0

    foreach ($currentProcess in @($Processes)) {
        try {
            [string]$processName = $currentProcess.ProcessName
            [int]$processId = $currentProcess.Id
            [DateTime]$processStartTime = $currentProcess.StartTime
            [double]$currentCpuSeconds = (
                $currentProcess.TotalProcessorTime.TotalSeconds
            )
            [int64]$currentWorkingSetBytes = (
                $currentProcess.WorkingSet64
            )
            [int64]$currentPrivateMemoryBytes = (
                $currentProcess.PrivateMemorySize64
            )
        }
        catch {
            # Exited or unreadable processes do not enter this live sample.
            continue
        }

        if (
            [string]::IsNullOrWhiteSpace($processName) -or
            [double]::IsNaN($currentCpuSeconds) -or
            [double]::IsInfinity($currentCpuSeconds) -or
            $currentCpuSeconds -lt 0.0 -or
            $currentWorkingSetBytes -lt 0 -or
            $currentPrivateMemoryBytes -lt 0
        ) {
            continue
        }

        $stableProcessKey = [string]::Format(
            [System.Globalization.CultureInfo]::InvariantCulture,
            "{0}|{1}|{2}",
            $processName.ToLowerInvariant(),
            $processId,
            $processStartTime.ToUniversalTime().Ticks
        )

        if (
            $LastCpuByProcess.ContainsKey($stableProcessKey) -and
            $currentCpuSeconds -lt (
                [double]$LastCpuByProcess[$stableProcessKey]
            )
        ) {
            # A lifetime CPU counter cannot fall for the same process key.
            continue
        }

        $LastCpuByProcess[$stableProcessKey] = $currentCpuSeconds
        $processCount++
        $workingSetBytes += $currentWorkingSetBytes
        $privateMemoryBytes += $currentPrivateMemoryBytes
    }

    [double]$cpuSeconds = 0.0

    foreach ($savedCpuSeconds in $LastCpuByProcess.Values) {
        $cpuSeconds += [double]$savedCpuSeconds
    }

    if ($cpuSeconds -lt $PreviousCpuTotal) {
        $cpuSeconds = $PreviousCpuTotal
    }

    return [pscustomobject]@{
        ProcessCount = $processCount
        WorkingSetMb = $workingSetBytes / 1MB
        PrivateMemoryMb = $privateMemoryBytes / 1MB
        CpuSeconds = $cpuSeconds
    }
}

function Get-SystemMemory {
    param(
        [Parameter(Mandatory = $true)]
        $ComputerInfo
    )

    try {
        $totalMemoryValue = $ComputerInfo.TotalPhysicalMemory
        $availableMemoryValue = $ComputerInfo.AvailablePhysicalMemory
    }
    catch {
        throw "Invalid system memory data: $($_.Exception.Message)"
    }

    if (
        $null -eq $totalMemoryValue -or
        $null -eq $availableMemoryValue -or
        $totalMemoryValue -is [bool] -or
        $availableMemoryValue -is [bool]
    ) {
        throw "Invalid system memory data: missing or boolean value."
    }

    [double]$totalMemoryMb = [double]$totalMemoryValue / 1MB
    [double]$availableMemoryMb = (
        [double]$availableMemoryValue / 1MB
    )

    if (
        [double]::IsNaN($totalMemoryMb) -or
        [double]::IsInfinity($totalMemoryMb) -or
        [double]::IsNaN($availableMemoryMb) -or
        [double]::IsInfinity($availableMemoryMb) -or
        $totalMemoryMb -le 0.0 -or
        $availableMemoryMb -lt 0.0 -or
        $availableMemoryMb -gt $totalMemoryMb
    ) {
        throw (
            "Invalid system memory data: " +
            "total=$totalMemoryMb, available=$availableMemoryMb."
        )
    }

    [double]$usedMemoryMb = (
        $totalMemoryMb - $availableMemoryMb
    )

    return [pscustomobject]@{
        TotalMemoryMb = $totalMemoryMb
        AvailableMemoryMb = $availableMemoryMb
        UsedMemoryMb = $usedMemoryMb
    }
}

if ($RunLabel.Contains("`r") -or $RunLabel.Contains("`n")) {
    throw "RunLabel must not contain newline characters."
}

$fullOutputPath = [System.IO.Path]::GetFullPath($OutputPath)

if (
    [System.IO.File]::Exists($fullOutputPath) -or
    [System.IO.Directory]::Exists($fullOutputPath)
) {
    throw "OutputPath already exists; refusing to overwrite it."
}

Add-Type -AssemblyName Microsoft.VisualBasic
$computerInfo = New-Object Microsoft.VisualBasic.Devices.ComputerInfo

$parentDirectory = [System.IO.Path]::GetDirectoryName($fullOutputPath)

if ([string]::IsNullOrWhiteSpace($parentDirectory)) {
    throw "OutputPath must have a valid parent directory."
}

[System.IO.Directory]::CreateDirectory($parentDirectory) | Out-Null

$header = @(
    "timestamp_utc",
    "elapsed_seconds",
    "run_label",
    "python_process_count",
    "python_working_set_mb",
    "python_private_memory_mb",
    "python_cpu_seconds",
    "ollama_process_count",
    "ollama_working_set_mb",
    "ollama_private_memory_mb",
    "ollama_cpu_seconds",
    "system_total_memory_mb",
    "system_available_memory_mb",
    "system_used_memory_mb"
) -join ","

$fileStream = $null
$writer = $null
$stopwatch = $null

try {
    $fileStream = [System.IO.File]::Open(
        $fullOutputPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::Read
    )
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    $writer = [System.IO.StreamWriter]::new(
        $fileStream,
        $utf8WithoutBom,
        4096,
        $false
    )
    $writer.WriteLine($header)
    $writer.Flush()

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    [double]$nextSampleMilliseconds = 0.0
    $pythonCpuByProcess = @{}
    $ollamaCpuByProcess = @{}
    [double]$pythonCpuTotal = 0.0
    [double]$ollamaCpuTotal = 0.0

    while ($true) {
        [double]$elapsedSeconds = $stopwatch.Elapsed.TotalSeconds

        if (
            $DurationSeconds -gt 0 -and
            $elapsedSeconds -ge $DurationSeconds
        ) {
            break
        }

        $timestampUtc = [DateTime]::UtcNow
        $allProcesses = @(Get-Process -ErrorAction SilentlyContinue)
        $pythonProcesses = @()
        $ollamaProcesses = @()

        foreach ($candidateProcess in $allProcesses) {
            try {
                [string]$candidateName = $candidateProcess.ProcessName
            }
            catch {
                continue
            }

            if (
                $candidateName -ieq "python" -or
                $candidateName -ieq "pythonw"
            ) {
                $pythonProcesses += $candidateProcess
            }

            if ($candidateName -match "(?i)(ollama|llama)") {
                $ollamaProcesses += $candidateProcess
            }
        }

        $python = Get-ProcessAggregate `
            -Processes $pythonProcesses `
            -LastCpuByProcess $pythonCpuByProcess `
            -PreviousCpuTotal $pythonCpuTotal
        $ollama = Get-ProcessAggregate `
            -Processes $ollamaProcesses `
            -LastCpuByProcess $ollamaCpuByProcess `
            -PreviousCpuTotal $ollamaCpuTotal
        $pythonCpuTotal = $python.CpuSeconds
        $ollamaCpuTotal = $ollama.CpuSeconds
        $systemMemory = Get-SystemMemory -ComputerInfo $computerInfo

        $fields = @(
            $timestampUtc.ToString(
                "o",
                [System.Globalization.CultureInfo]::InvariantCulture
            ),
            (Format-InvariantNumber -Value $elapsedSeconds),
            (ConvertTo-CsvField -Value $RunLabel),
            (Format-InvariantInteger -Value $python.ProcessCount),
            (Format-InvariantNumber -Value $python.WorkingSetMb),
            (Format-InvariantNumber -Value $python.PrivateMemoryMb),
            (Format-InvariantNumber -Value $python.CpuSeconds),
            (Format-InvariantInteger -Value $ollama.ProcessCount),
            (Format-InvariantNumber -Value $ollama.WorkingSetMb),
            (Format-InvariantNumber -Value $ollama.PrivateMemoryMb),
            (Format-InvariantNumber -Value $ollama.CpuSeconds),
            (Format-InvariantNumber -Value $systemMemory.TotalMemoryMb),
            (Format-InvariantNumber -Value $systemMemory.AvailableMemoryMb),
            (Format-InvariantNumber -Value $systemMemory.UsedMemoryMb)
        )

        $writer.WriteLine($fields -join ",")
        $writer.Flush()

        $nextSampleMilliseconds += [double]$IntervalMilliseconds
        [double]$sleepMilliseconds = (
            $nextSampleMilliseconds - $stopwatch.Elapsed.TotalMilliseconds
        )

        if ($sleepMilliseconds -gt 0.0) {
            Start-Sleep -Milliseconds (
                [int][Math]::Ceiling($sleepMilliseconds)
            )
        }
    }
}
finally {
    if ($null -ne $stopwatch) {
        $stopwatch.Stop()
    }

    if ($null -ne $writer) {
        try {
            $writer.Flush()
        }
        finally {
            $writer.Dispose()
        }
    }
    elseif ($null -ne $fileStream) {
        $fileStream.Dispose()
    }
}
