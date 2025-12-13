# mocnpub Benchmark Script
# Runs for a specified number of seconds and measures keys/sec
#
# Usage:
#   .\scripts\benchmark.ps1
#   .\scripts\benchmark.ps1 -Seconds 120 -BatchSize 3584000
#   .\scripts\benchmark.ps1 -KeysPerThread 2048  # Rebuild and run
#
# When KeysPerThread is changed, it sets the MAX_KEYS_PER_THREAD environment
# variable and runs cargo build --release before starting the benchmark.

param(
    [int]$Seconds = 120,
    [int]$BatchSize = 3584000,
    [int]$ThreadsPerBlock = 128,
    [int]$KeysPerThread = 1408,
    [string]$Prefix = "00000000",
    [switch]$SkipBuild = $false
)

$projectRoot = Join-Path $PSScriptRoot ".."
$projectRoot = [System.IO.Path]::GetFullPath($projectRoot)
$exePath = Join-Path $projectRoot "target\release\mocnpub-main.exe"

Write-Host "=== mocnpub Benchmark ===" -ForegroundColor Cyan
Write-Host "Parameters:" -ForegroundColor Yellow
Write-Host "  BatchSize:        $BatchSize"
Write-Host "  ThreadsPerBlock:  $ThreadsPerBlock"
Write-Host "  KeysPerThread:    $KeysPerThread (build-time)"
Write-Host "  Duration:         $Seconds sec"
Write-Host "  Prefix:           $Prefix"
Write-Host ""

# Build (unless -SkipBuild is specified)
if (-not $SkipBuild) {
    Write-Host "Building with MAX_KEYS_PER_THREAD=$KeysPerThread..." -ForegroundColor Yellow
    $env:MAX_KEYS_PER_THREAD = $KeysPerThread
    Push-Location $projectRoot
    try {
        cargo build --release
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: cargo build failed" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
        Remove-Item Env:\MAX_KEYS_PER_THREAD -ErrorAction SilentlyContinue
    }
    Write-Host "Build completed!" -ForegroundColor Green
    Write-Host ""
}

if (-not (Test-Path $exePath)) {
    Write-Host "Error: $exePath not found." -ForegroundColor Red
    exit 1
}

# Start process
$procArgs = @(
    "mine",
    "--gpu",
    "--prefix", $Prefix,
    "--limit", "0",
    "--batch-size", $BatchSize,
    "--threads-per-block", $ThreadsPerBlock
)

$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $exePath
$pinfo.Arguments = $procArgs -join " "
$pinfo.RedirectStandardOutput = $true
$pinfo.RedirectStandardError = $true
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $pinfo

Write-Host "Starting benchmark..." -ForegroundColor Green
$process.Start() | Out-Null

# Wait for specified seconds
Start-Sleep -Seconds $Seconds

# Terminate process
$process.Kill()
$process.WaitForExit()

# Get output
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()

# Extract the last keys/sec value
$lines = $stdout -split "`n"
$lastKeysPerSec = $null

foreach ($line in $lines) {
    if ($line -match "(\d+\.?\d*)\s+keys/sec") {
        $lastKeysPerSec = $matches[1]
    }
}

Write-Host ""
Write-Host "=== Result ===" -ForegroundColor Cyan
if ($lastKeysPerSec) {
    $keysPerSecFloat = [double]$lastKeysPerSec
    $keysPerSecB = $keysPerSecFloat / 1e9
    Write-Host "Keys/sec: $lastKeysPerSec ($([math]::Round($keysPerSecB, 3))B)" -ForegroundColor Green

    # CSV 形式で出力（コピペ用）
    Write-Host ""
    Write-Host "CSV: $BatchSize,$ThreadsPerBlock,$KeysPerThread,$lastKeysPerSec" -ForegroundColor Yellow
} else {
    Write-Host "Could not extract keys/sec from output" -ForegroundColor Red
    Write-Host "Output:" -ForegroundColor Yellow
    Write-Host $stdout
}
