param(
    [string]$TrainData = "data/external/ultrachat_train_sft.txt",
    [string]$ValidationData = "data/external/ultrachat_test_sft.txt",
    [string]$Checkpoint = "models/ultron_ultrachat_control.bin",
    [int]$Epochs = 1,
    [double]$LearningRate = 0.001,
    [int]$Speed = 10,
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
$buildDirectory = Join-Path $repoRoot "build"
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
$ninjaCommand = Get-Command ninja -ErrorAction SilentlyContinue

if ($null -ne $cmakeCommand) {
    cmake --build $buildDirectory --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "Release build failed." }
} elseif ($null -ne $ninjaCommand -and (Test-Path (Join-Path $buildDirectory "build.ninja"))) {
    $jobs = [Math]::Max(1, [Environment]::ProcessorCount)
    Write-Host "CMake is not on PATH; building with Ninja using $jobs logical processors."
    ninja -C $buildDirectory -j $jobs
    if ($LASTEXITCODE -ne 0) { throw "Ninja build failed." }
} elseif (Test-Path (Join-Path $buildDirectory "Release\ultron.exe")) {
    Write-Host "CMake/Ninja not found, but an existing Release executable is available. Skipping rebuild."
} elseif (Test-Path (Join-Path $buildDirectory "ultron.exe")) {
    Write-Host "CMake/Ninja not found, but an existing executable is available. Skipping rebuild."
} else {
    throw "Neither CMake nor Ninja is available, and no existing ULTRON executable was found. Install CMake or add it to PATH."
}

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
if ($Speed -eq 10) {
    Write-Host " Mode       : TURBO (16-token context / 8-token sampling stride)"
}
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

if ($exitCode -eq 10) {
    Write-Host ""
    Write-Host "Ctrl+T requested a safe stop and ULTRON test mode was completed."
    Write-Host "The saved checkpoint is preserved. Skipping full-corpus post-training evaluation."
    exit 0
}

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
