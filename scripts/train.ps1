param(
    [string]$Data = "data/train.txt",
    [int]$Epochs = 20,
    [double]$LearningRate = 0.001,
    [string]$Checkpoint = "models/ultron_bpe.bin",
    [string]$ValidationData = "data/validation.txt",
    [int]$SaveEvery = 10,
    [switch]$ExternalCorpus,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON build failed with exit code $LASTEXITCODE."
}

ctest --test-dir build --output-on-failure

$ctestExitCode = $LASTEXITCODE

$exe = "build/Release/ultron.exe"

if ($ExternalCorpus) {
    $Data = "data/external/tatoeba_en_cc0.txt"
    if (!(Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) $Data))) {
        throw "External corpus not found. Run scripts/download_tatoeba.ps1 first."
    }
}
if (!(Test-Path $exe)) { $exe = "build/ultron.exe" }
if (!(Test-Path $exe)) { throw "ULTRON executable not found. Run scripts/build.ps1 first." }

if ($ctestExitCode -ne 0) {
    Write-Host "CTest could not complete. Running the executable self-test directly..."
    & $exe --self-test

    if ($LASTEXITCODE -ne 0) {
        throw "ULTRON smoke tests failed. CTest exit code $ctestExitCode; direct self-test exit code $LASTEXITCODE. Training was not started."
    }

    Write-Host "Direct ULTRON self-test passed."
} else {
    Write-Host "CTest smoke tests passed."
}

$dataPath = Join-Path (Split-Path -Parent $PSScriptRoot) $Data
if (!(Test-Path $dataPath)) {
    throw "Training corpus not found: $Data"
}

$dataInfo = Get-Item $dataPath
$lineCount = (Get-Content $dataPath | Measure-Object -Line).Lines
Write-Host ("Training corpus: {0} bytes, {1} lines" -f $dataInfo.Length, $lineCount)

if ($dataInfo.Length -lt 20000) {
    throw "Training corpus is unexpectedly small. Refusing to start training."
}

if (!$ExternalCorpus) {
    $rawText = Get-Content $dataPath -Raw
    foreach ($required in @("USER:", "ULTRON:", "Hello,", "I'm", "?")) {
        if ($rawText -notlike "*$required*") {
            throw "Training corpus is missing expected language pattern: $required"
        }
    }
}

$arguments = @(

    "--train", $Data,
    "--epochs", $Epochs,
    "--lr", $LearningRate,
    "--save", $Checkpoint,
    "--save-every", $SaveEvery,
    "--non-interactive"
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

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON TRAINING STARTED"
Write-Host " Epochs: $Epochs | Learning rate: $LearningRate"
Write-Host " Checkpoint: $Checkpoint"
Write-Host " Watch for [train] messages below."
Write-Host "============================================================"
Write-Host ""

$trainingStart = Get-Date
& $exe @arguments
$trainingEnd = Get-Date

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON training failed with exit code $LASTEXITCODE."
}

$trainingDuration = $trainingEnd - $trainingStart
Write-Host ""
Write-Host ("ULTRON training process finished in {0:hh\\:mm\\:ss}" -f $trainingDuration)
Write-Host "Now checking the saved checkpoint..."
Write-Host ""

$evaluationArgs = @(
    "--load", $Checkpoint,
    "--eval", $Data,
    "--no-online-learning",
    "--non-interactive"
)

$validationPath = Join-Path (Split-Path -Parent $PSScriptRoot) $ValidationData
if (Test-Path $validationPath) {
    Write-Host ""
    Write-Host "Evaluating held-out validation corpus..."
    & $exe --load $Checkpoint --eval $ValidationData --no-online-learning --non-interactive

    if ($LASTEXITCODE -ne 0) {
        throw "ULTRON validation evaluation failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Evaluating saved checkpoint..."
& $exe @evaluationArgs

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON checkpoint evaluation failed with exit code $LASTEXITCODE."
}

Write-Host "Training and checkpoint evaluation finished successfully."
