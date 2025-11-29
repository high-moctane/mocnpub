# mocnpub ベンチマークスクリプト
# 指定した秒数だけ実行して、keys/sec を取得する
#
# 使い方:
#   .\scripts\benchmark.ps1 -Seconds 60
#   .\scripts\benchmark.ps1 -Seconds 60 -ThreadsPerBlock 128 -KeysPerThread 256
#   .\scripts\benchmark.ps1 -Seconds 60 -BatchSize 131072

param(
    [int]$Seconds = 60,
    [int]$BatchSize = 65536,
    [int]$ThreadsPerBlock = 64,
    [int]$KeysPerThread = 256,
    [string]$Prefix = "0000"
)

$exePath = ".\target\release\mocnpub-main.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "Error: $exePath not found. Run 'cargo build --release' first." -ForegroundColor Red
    exit 1
}

Write-Host "=== mocnpub Benchmark ===" -ForegroundColor Cyan
Write-Host "Parameters:" -ForegroundColor Yellow
Write-Host "  BatchSize:        $BatchSize"
Write-Host "  ThreadsPerBlock:  $ThreadsPerBlock"
Write-Host "  KeysPerThread:    $KeysPerThread"
Write-Host "  Duration:         $Seconds sec"
Write-Host "  Prefix:           $Prefix"
Write-Host ""

# プロセスを起動
$procArgs = @(
    "--gpu",
    "--prefix", $Prefix,
    "--limit", "0",
    "--batch-size", $BatchSize,
    "--threads-per-block", $ThreadsPerBlock,
    "--keys-per-thread", $KeysPerThread
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

# 指定秒数待つ
Start-Sleep -Seconds $Seconds

# プロセスを終了
$process.Kill()
$process.WaitForExit()

# 出力を取得
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()

# 最後の keys/sec を抽出
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
