param(
    [string]$Data = "data/train.txt",
    [int]$Epochs = 5,
    [double]$LearningRate = 0.003,
    [string]$Checkpoint = "models/ultron.bin",
    [switch]$ExternalCorpus
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

& $exe --train $Data --epochs $Epochs --lr $LearningRate --save $Checkpoint