param(
    [string]$Checkpoint = "models/ultron_ultrachat_control.bin",
    [string]$DialogueData = "data/external/ultrachat_train_sft.txt",
    [int]$EnglishSentences = 15000,
    [int]$EnglishChars = 750000,
    [int]$DialogueChars = 250000,
    [int]$Epochs = 3,
    [double]$LearningRate = 0.0003,
    [int]$Speed = 3,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build/Release/ultron.exe"
if (!(Test-Path $exe)) { $exe = Join-Path $repoRoot "build/ultron.exe" }
if (!(Test-Path $exe)) {
    throw "ULTRON executable not found. Run .\scripts\build.ps1 first."
}

if ($EnglishSentences -lt 100) { throw "EnglishSentences must be at least 100." }
if ($EnglishChars -lt 10000) { throw "EnglishChars is too small." }
if ($DialogueChars -lt 10000) { throw "DialogueChars is too small." }
if ($Epochs -lt 1) { throw "Epochs must be at least 1." }
if ($LearningRate -le 0) { throw "LearningRate must be positive." }
if ($Speed -lt 1 -or $Speed -gt 10) { throw "Speed must be between 1 and 10." }

$checkpointPath = Join-Path $repoRoot $Checkpoint
$dialoguePath = Join-Path $repoRoot $DialogueData
$englishPath = Join-Path $repoRoot "data/external/tatoeba_en_cc0.txt"
$mixedPath = Join-Path $repoRoot "data/external/ultron_english_mixed.txt"

if ($Fresh) {
    Remove-Item $checkpointPath -Force -ErrorAction SilentlyContinue
}

if (!(Test-Path $dialoguePath)) {
    throw "UltraChat dialogue corpus not found: $DialogueData. Run .\scripts\download_ultrachat.ps1 first."
}

if (!(Test-Path $englishPath)) {
    Write-Host "English CC0 corpus not found. Downloading it now..."
    & (Join-Path $repoRoot "scripts/download_tatoeba.ps1") -MaxSentences $EnglishSentences
    if ($LASTEXITCODE -ne 0) {
        throw "English corpus download failed with exit code $LASTEXITCODE."
    }
}

if (!(Test-Path $englishPath)) {
    throw "English corpus was not created: $englishPath"
}

if (!(Test-Path $checkpointPath)) {
    throw "Checkpoint not found: $Checkpoint. Train ULTRON first, or use -Fresh to intentionally start from a new English-trained model."
}

function Get-TextSample {
    param(
        [string]$Path,
        [int]$MaxChars
    )

    $lines = @(Get-Content -Path $Path)
    if ($lines.Count -eq 0) { return "" }

    $builder = [System.Text.StringBuilder]::new()
    $maxAttempts = [Math]::Max(20, [Math]::Ceiling($MaxChars / 1000))

    for ($attempt = 0; $attempt -lt $maxAttempts; $attempt++) {
        $lineIndex = Get-Random -Minimum 0 -Maximum $lines.Count
        $line = [string]$lines[$lineIndex]

        if ([string]::IsNullOrWhiteSpace($line)) { continue }

        $candidate = $line.Trim()
        if (($builder.Length + $candidate.Length + 1) -gt $MaxChars) {
            continue
        }

        [void]$builder.AppendLine($candidate)

        if ($builder.Length -ge $MaxChars -or $attempt -ge ($maxAttempts - 1)) {
            break
        }
    }

    return $builder.ToString()
}

function Get-EnglishSample {
    param(
        [string]$Path,
        [int]$SentenceCount,
        [int]$MaxChars
    )

    $lines = @(Get-Content -Path $Path | Where-Object { ![string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -eq 0) { throw "English corpus is empty: $Path" }

    $count = [Math]::Min($SentenceCount, $lines.Count)
    $selected = if ($count -lt $lines.Count) {
        @(Get-Random -InputObject $lines -Count $count)
    } else {
        $lines
    }

    $builder = [System.Text.StringBuilder]::new()

    foreach ($line in $selected) {
        $sentence = ([string]$line).Trim()
        if ($sentence.Length -eq 0) { continue }

        if (($builder.Length + $sentence.Length + 1) -gt $MaxChars) {
            break
        }

        [void]$builder.AppendLine($sentence)
    }

    return $builder.ToString()
}

$englishSample = Get-EnglishSample -Path $englishPath -SentenceCount $EnglishSentences -MaxChars $EnglishChars
$dialogueSample = Get-TextSample -Path $dialoguePath -MaxChars $DialogueChars

if ([string]::IsNullOrWhiteSpace($englishSample)) {
    throw "English sample is empty."
}

if ([string]::IsNullOrWhiteSpace($dialogueSample)) {
    throw "Dialogue sample is empty."
}

$header = @(
    "# ULTRON mixed language corpus",
    "# English sentences are learned from the external CC0 corpus.",
    "# Dialogue examples are learned from the downloaded UltraChat control corpus.",
    "# No model answers are hard-coded here.",
    ""
) -join [Environment]::NewLine

$mixedText = $header + $englishSample.TrimEnd() + [Environment]::NewLine + [Environment]::NewLine + $dialogueSample.Trim()
[System.IO.File]::WriteAllText(
    $mixedPath,
    $mixedText,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON LEARNED-ENGLISH TRAINING"
Write-Host " Checkpoint : $Checkpoint"
Write-Host " English    : $EnglishSentences sentences / $EnglishChars chars"
Write-Host " Dialogue   : $DialogueChars chars"
Write-Host " Epochs     : $Epochs"
Write-Host " Learning   : $LearningRate"
Write-Host " Speed      : $Speed/10"
Write-Host " Hotkeys    : Ctrl+U = save | Ctrl+T = save + stop + test"
Write-Host "============================================================"
Write-Host "No hard-coded answers are added; language is learned from data."
Write-Host ""

$args = @(
    "--load", $Checkpoint,
    "--train", $mixedPath,
    "--epochs", $Epochs,
    "--lr", $LearningRate,
    "--save", $Checkpoint,
    "--save-every", 1,
    "--speed", $Speed,
    "--non-interactive"
)

& $exe @args
$exitCode = $LASTEXITCODE

if ($exitCode -eq 10) {
    Write-Host ""
    Write-Host "Ctrl+T safely stopped training after saving and completed the interactive test."
    exit 0
}

if ($exitCode -ne 0) {
    throw "ULTRON English training failed with exit code $exitCode."
}

Write-Host ""
Write-Host "Learned-English training finished and checkpoint saved:"
Write-Host $Checkpoint
