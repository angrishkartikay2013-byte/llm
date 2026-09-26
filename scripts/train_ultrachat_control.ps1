param(
    [string]$TrainData = "data/external/ultrachat_train_sft.txt",
    [string]$ValidationData = "data/external/ultrachat_test_sft.txt",
    [string]$Checkpoint = "models/ultron_ultrachat_control.bin",
    [int]$Epochs = 5,
    [double]$LearningRate = 0.001,
    [int]$Speed = 1,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$trainPath = Join-Path $repoRoot $TrainData
$validationPath = Join-Path $repoRoot $ValidationData
$checkpointPath = Join-Path $repoRoot $Checkpoint

if ($Speed -lt 1 -or $Speed -gt 10) { throw "Speed must be between 1 and 10." }
if ($Epochs -lt 1) { throw "Epochs must be at least 1." }
if (!(Test-Path $trainPath)) { throw "HF training corpus not found. Run .\scripts\download_ultrachat.ps1 first." }
if (!(Test-Path $validationPath)) { throw "HF validation corpus not found. Run .\scripts\download_ultrachat.ps1 first." }

$exe = Join-Path $repoRoot "build/Release/ultron.exe"
if (!(Test-Path $exe)) { $exe = Join-Path $repoRoot "build/ultron.exe" }
if (!(Test-Path $exe)) { throw "ULTRON executable not found. Build it first." }

Write-Host "Building ULTRON Release..."
cmake --build (Join-Path $repoRoot "build") --config Release
if ($LASTEXITCODE -ne 0) { throw "Release build failed." }

Write-Host "Running self-test..."
& $exe --self-test
if ($LASTEXITCODE -ne 0) { throw "ULTRON self-test failed. Training was not started." }

if ($Fresh) {
    Remove-Item $checkpointPath -Force -ErrorAction SilentlyContinue
}

$trainInfo = Get-Item $trainPath
$validationInfo = Get-Item $validationPath

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON HF CONTROL EXPERIMENT"
Write-Host " Train      : $TrainData ($($trainInfo.Length) bytes)"
Write-Host " Validation : $ValidationData ($($validationInfo.Length) bytes)"
Write-Host " Epochs     : $Epochs"
Write-Host " Learning   : $LearningRate"
Write-Host " Speed      : $Speed/10"
Write-Host " Checkpoint : $Checkpoint"
Write-Host "============================================================"

$arguments = @(
    "--train", $TrainData,
    "--epochs", $Epochs,
    "--lr", $LearningRate,
    "--save", $Checkpoint,
    "--save-every", 1,
    "--speed", $Speed,
    "--non-interactive"
)

if ((Test-Path $checkpointPath) -and !$Fresh) {
    $arguments += @("--load", $Checkpoint)
    Write-Host "Resuming existing HF control checkpoint."
} else {
    Write-Host "Starting a completely fresh HF control model."
}

$started = Get-Date
& $exe @arguments
$exitCode = $LASTEXITCODE
$finished = Get-Date

if ($exitCode -ne 0) { throw "HF control training failed with exit code $exitCode." }
if (!(Test-Path $checkpointPath)) { throw "Training reported success but did not create $Checkpoint." }

Write-Host ""
Write-Host ("Training finished in {0:hh\:mm\:ss}" -f ($finished - $started))
Write-Host ""
Write-Host "================ HF TRAIN METRICS ================"
& $exe --load $Checkpoint --eval $TrainData --no-online-learning --non-interactive
if ($LASTEXITCODE -ne 0) { throw "HF training-set evaluation failed." }

Write-Host ""
Write-Host "=============== HF TEST METRICS ================="
& $exe --load $Checkpoint --eval $ValidationData --no-online-learning --non-interactive
if ($LASTEXITCODE -ne 0) { throw "HF held-out evaluation failed." }

Write-Host ""
Write-Host "============================================================"
Write-Host " HF CONTROL EXPERIMENT FINISHED"
Write-Host "============================================================"
