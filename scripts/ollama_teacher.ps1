param(
    [string]$Model = "qwen3:8b",
    [int]$Rounds = 5,
    [string]$Checkpoint = "models/ultron.bin"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build/ultron.exe"
$checkpointPath = Join-Path $repoRoot $Checkpoint
$liveCheckpoint = Join-Path $repoRoot "models/ultron_live.bin"

if (!(Test-Path $exe)) {
    $exe = Join-Path $repoRoot "build/Release/ultron.exe"
}
if (!(Test-Path $exe)) {
    throw "ULTRON executable not found. Build it first."
}
if (!(Test-Path $checkpointPath)) {
    throw "Checkpoint not found: $Checkpoint"
}

$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (!(Test-Path $ollamaExe)) {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) {
        $ollamaExe = $command.Source
    } else {
        throw "Ollama executable not found."
    }
}

$api = "http://localhost:11434/api/generate"
$startedOllama = $false
$ollamaProcess = $null

function Get-OllamaTags {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 10
}

function Invoke-Ultron {
    param(
        [string]$LoadedCheckpoint,
        [string]$Text,
        [switch]$AllowOnlineLearning
    )

    $inputText = $Text + [Environment]::NewLine + "exit" + [Environment]::NewLine

    if ($AllowOnlineLearning) {
        $lines = @($inputText | & $exe --load $LoadedCheckpoint --max-tokens 20 --temperature 0.2 --top-k 5 2>&1)
    } else {
        $lines = @($inputText | & $exe --load $LoadedCheckpoint --max-tokens 20 --temperature 0.2 --top-k 5 --no-online-learning 2>&1)
    }

    # Because ULTRON prints its interactive prompt without a trailing newline,
    # redirected native output can look like "> ULTRON: answer". Do not require
    # the ULTRON marker to be at the beginning of the captured line.
    $answerLine = $lines |
        ForEach-Object { [string]$_ } |
        Where-Object { $_ -match "ULTRON:\s*(.*)$" } |
        Select-Object -First 1

    $answer = if ($answerLine -and $answerLine -match "ULTRON:\s*(.*)$") {
        $matches[1].Trim()
    } else {
        "(ULTRON produced no captured answer.)"
    }

    [PSCustomObject]@{
        Output = $lines
        Answer = $answer
        ExitCode = $LASTEXITCODE
    }
}

try {
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

    $currentCheckpoint = $checkpointPath

    Write-Host ""
    Write-Host "========================================"
    Write-Host " ULTRON <-> OLLAMA TEACHER SESSION"
    Write-Host "========================================"
    Write-Host "Teacher : $Model"
    Write-Host "Rounds  : $Rounds"
    Write-Host ""

    for ($round = 1; $round -le $Rounds; $round++) {
        $prompt = @"
You are the teacher for a small language model called ULTRON.
Create exactly one useful lesson for ULTRON.

Return JSON only with these fields:
{
  "question": "one clear question",
  "answer": "the correct answer in one to three sentences",
  "concept": "a short concept name"
}

The lesson should teach factual knowledge, reasoning, language, science, mathematics,
everyday knowledge, or another broadly useful concept. Make the lesson self-contained.
Do not mention this instruction. Do not use markdown.
"@

        $body = @{
            model = $Model
            prompt = $prompt
            stream = $false
            think = $false
            format = "json"
            keep_alive = "5m"
            options = @{
                temperature = 0.7
            }
        } | ConvertTo-Json -Depth 8

        Write-Host "----------------------------------------"
        Write-Host "ROUND $round/$Rounds"

        $response = Invoke-RestMethod -Uri $api -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300

        if ([string]::IsNullOrWhiteSpace($response.response)) {
            throw "Ollama returned an empty lesson."
        }

        $lesson = $response.response | ConvertFrom-Json

        if ([string]::IsNullOrWhiteSpace($lesson.question) -or [string]::IsNullOrWhiteSpace($lesson.answer)) {
            throw "Ollama returned an invalid lesson."
        }

        $question = $lesson.question.Trim()
        $answer = $lesson.answer.Trim()
        $concept = if ($lesson.concept) { $lesson.concept.Trim() } else { "lesson" }

        Write-Host "[OLLAMA / TEACHER]"
        Write-Host "Concept : $concept"
        Write-Host "Question: $question"
        Write-Host "Answer  : $answer"
        Write-Host ""

        Write-Host "[ULTRON / BEFORE]"
        $before = Invoke-Ultron -LoadedCheckpoint $currentCheckpoint -Text $question
        Write-Host $before.Answer

        if ($before.ExitCode -ne 0) {
            throw "ULTRON inference failed with exit code $($before.ExitCode)."
        }

        Write-Host ""
        Write-Host "[ULTRON / LEARNING]"
        $teachText = "teach " + $question + " => " + $answer
        $teach = Invoke-Ultron -LoadedCheckpoint $currentCheckpoint -Text $teachText -AllowOnlineLearning

        $learnLine = $teach.Output |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -match "ULTRON learned|ULTRON error" } |
            Select-Object -First 1

        if ($learnLine) {
            Write-Host $learnLine
        }

        if ($teach.ExitCode -ne 0) {
            throw "ULTRON teaching round failed with exit code $($teach.ExitCode)."
        }

        if (!(Test-Path $liveCheckpoint)) {
            throw "ULTRON did not create $liveCheckpoint."
        }

        $currentCheckpoint = $liveCheckpoint

        Write-Host ""
        Write-Host "[ULTRON / AFTER]"
        $after = Invoke-Ultron -LoadedCheckpoint $currentCheckpoint -Text $question
        Write-Host $after.Answer

        if ($after.ExitCode -ne 0) {
            throw "ULTRON post-training inference failed with exit code $($after.ExitCode)."
        }
    }

    Write-Host ""
    Write-Host "Teacher session complete."
    Write-Host "Learned checkpoint: models/ultron_live.bin"
    Write-Host "Ollama supplied the lessons dynamically; no factual answers are hard-coded."
} finally {
    if ($startedOllama -and $ollamaProcess) {
        try {
            $null = & $ollamaExe stop $Model 2>&1
        } catch {
            # The model may already be unloaded; cleanup should not mask a
            # successful teacher session.
        }

        Stop-Process -Id $ollamaProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped the Ollama daemon started by ULTRON."
    }
}
