param(
    [string]$Model = "qwen3:8b",
    [int]$Batches = 1,
    [int]$SentencesPerBatch = 30,
    [int]$Epochs = 1,
    [double]$LearningRate = 0.003,
    [string]$Output = "data/external/ollama_distill.txt",
    [string]$Checkpoint = "models/ultron_bpe.bin",
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoRoot $Output
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$api = "http://localhost:11434/api/generate"
$startedOllama = $false
$ollamaProcess = $null

if (!(Test-Path $ollamaExe)) {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) {
        $ollamaExe = $command.Source
    } else {
        throw "Ollama executable not found. Start/install Ollama first."
    }
}

function Get-OllamaTags {
    return Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 10
}

try {
    $tags = Get-OllamaTags
} catch {
    Write-Host "Ollama daemon is not running. Starting it..."
    $ollamaProcess = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
    $startedOllama = $true

    $connected = $false

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        Start-Sleep -Seconds 1

        try {
            $tags = Get-OllamaTags
            $connected = $true
            break
        } catch {
            # Keep waiting for the local daemon to start.
        }
    }

    if (!$connected) {
        throw "Ollama could not be started at localhost:11434."
    }
}

$available = @($tags.models | ForEach-Object { $_.name })
if ($available -notcontains $Model) {
    throw "Model '$Model' was not found. Run 'ollama list' and pass -Model with an installed model."
}

$seedPath = Join-Path $repoRoot "data/train.txt"
$conversationPath = Join-Path $repoRoot "data/conversations.txt"
$seed = Get-Content $seedPath -Raw

$conversation = ""
if (Test-Path $conversationPath) {
    $conversation = Get-Content $conversationPath -Raw
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null

$writer = [System.IO.StreamWriter]::new(
    $outputPath,
    $false,
    [System.Text.UTF8Encoding]::new($false)
)

try {
    for ($batch = 1; $batch -le $Batches; $batch++) {
        $prompt = @"
You are generating training text for a small language model called ULTRON.
Return exactly $SentencesPerBatch plain English sentences, one sentence per line.
Do not number them. Do not add headings. Do not explain the task.
Mix factual explanations, simple dialogue, question-and-answer patterns, definitions,
short descriptions, reasoning examples, and everyday English.
Keep each sentence under about 20 words.

Existing seed material:
$seed

Recent local conversation memory, if any:
$conversation
"@

        $body = @{
            model = $Model
            prompt = $prompt
            stream = $false
            think = $false
            keep_alive = "5m"
            options = @{
                temperature = 0.7
            }
        } | ConvertTo-Json -Depth 6

        Write-Host "Ollama batch $batch/$Batches..."

        try {
            $response = Invoke-RestMethod -Uri $api -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
        } catch {
            throw "Ollama generation failed: $($_.Exception.Message)"
        }

        if ([string]::IsNullOrWhiteSpace($response.response)) {
            throw "Ollama returned an empty response."
        }

        foreach ($line in ($response.response -split "`r?`n")) {
            $clean = $line.Trim()
            $clean = $clean -replace '^[\-*]+\s*', ''
            $clean = $clean -replace '^\d+[\.)]\s*', ''
            if ($clean.Length -gt 0) {
                $writer.WriteLine($clean)
            }
        }
    }
} finally {
    $writer.Dispose()
}

$ollamaTrainData = Get-Content $outputPath -Raw
$combinedPath = Join-Path $repoRoot "data/external/ollama_combined_train.txt"
$seed + "`n" + $ollamaTrainData | Set-Content -Path $combinedPath -Encoding utf8

$exe = Join-Path $repoRoot "build/ultron.exe"
if (!(Test-Path $exe)) {
    $exe = Join-Path $repoRoot "build/Release/ultron.exe"
}
if (!(Test-Path $exe)) {
    throw "ULTRON executable not found. Build it first."
}

Write-Host "Training ULTRON on the Ollama-generated corpus..."

$checkpointPath = Join-Path $repoRoot $Checkpoint
$arguments = @(
    "--train", $combinedPath,
    "--epochs", $Epochs,
    "--lr", $LearningRate,
    "--save", $Checkpoint
)

if ((Test-Path $checkpointPath) -and !$Fresh) {
    Write-Host "Resuming from checkpoint: $Checkpoint"
    $arguments += @("--load", $Checkpoint)
} elseif ($Fresh) {
    Write-Host "Starting a fresh ULTRON model."
} else {
    Write-Host "No checkpoint found; starting a fresh ULTRON model."
}

& $exe @arguments

if ($LASTEXITCODE -ne 0) {
    throw "ULTRON training failed with exit code $LASTEXITCODE."
}

Write-Host "Unloading Ollama model..."
& $ollamaExe stop $Model | Out-Null

if ($startedOllama -and $ollamaProcess) {
    Stop-Process -Id $ollamaProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped the Ollama daemon started by ULTRON."
}

Write-Host "Done. ULTRON now has a local Ollama-distilled training corpus."
Write-Host "Ollama was used only through localhost and is unloaded after the bootstrap."