param(
    [string]$Data = "data/train.txt",
    [int]$Epochs = 5,
    [double]$LearningRate = 0.001,
    [string]$Checkpoint = "models/ultron_bpe.bin",
    [int]$SaveEvery = 10,
    [switch]$ExternalCorpus,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

cmake --build build --config Release

$exe = "build/Release/ultron.exe"

if ($ExternalCorpus) {
    $Data = "data/external/tatoeba_en_cc0.txt"
    if (!(Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) $Data))) {
        throw "External corpus not found. Run scripts/download_tatoeba.ps1 first."
    }
}
if (!(Test-Path $exe)) { $exe = "build/ultron.exe" }
if (!(Test-Path $exe)) { throw "ULTRON executable not found. Run scripts/build.ps1 first." }

$arguments = @(
    "--train", $Data,
    "--epochs", $Epochs,
    "--lr", $LearningRate,
    "--save", $Checkpoint,
    "--save-every", $SaveEvery
)

$checkpointPath = Join-Path (Split-Path -Parent $PSScriptRoot) $Checkpoint

if ((Test-Path $checkpointPath) -and !$Fresh) {
    Write-Host "Resuming from checkpoint: $Checkpoint"
    $arguments += @("--load", $Checkpoint)
} elseif ($Fresh) {
    Write-Host "Starting a fresh model."
} else {
    Write-Host "No existing checkpoint found; starting a fresh model."
}

& $exe @arguments

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON training failed with exit code $LASTEXITCODE."
}

$evaluationArgs = @(
    "--load", $Checkpoint,
    "--eval", $Data,
    "--no-online-learning"
)

Write-Host ""
Write-Host "Evaluating saved checkpoint..."
& $exe @evaluationArgs

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON checkpoint evaluation failed with exit code $LASTEXITCODE."
}

Write-Host "Training and checkpoint evaluation finished successfully."
