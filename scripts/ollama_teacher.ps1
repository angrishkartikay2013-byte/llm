param(
    [string]$Model = "qwen3:8b",
    [int]$Rounds = 5,
    [int]$ReplayEpochs = 5,
    [float]$LearningRate = 0.001,
    [string]$Checkpoint = "models/ultron_bpe.bin",
    [switch]$ResetTeacherCorpus
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build/ultron.exe"
$checkpointPath = Join-Path $repoRoot $Checkpoint
$liveCheckpoint = Join-Path $repoRoot "models/ultron_live.bin"

$baseCorpusPath = Join-Path $repoRoot "data/train.txt"
$teacherCorpusPath = Join-Path $repoRoot "data/external/ollama_teacher.txt"
$combinedCorpusPath = Join-Path $repoRoot "data/external/ollama_teacher_combined.txt"

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data/external") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "models") | Out-Null

if (!(Test-Path $exe)) {
    $exe = Join-Path $repoRoot "build/Release/ultron.exe"
}
if (!(Test-Path $exe)) {
    throw "ULTRON executable not found. Build it first."
}
if (!(Test-Path $checkpointPath)) {
    throw "Checkpoint not found: $Checkpoint"
}
if (!(Test-Path $baseCorpusPath)) {
    throw "Base training corpus not found: data/train.txt"
}

if ($ResetTeacherCorpus) {
    Remove-Item $teacherCorpusPath -Force -ErrorAction SilentlyContinue
    Remove-Item $combinedCorpusPath -Force -ErrorAction SilentlyContinue
    Remove-Item $liveCheckpoint -Force -ErrorAction SilentlyContinue
}

if (!(Test-Path $teacherCorpusPath)) {
    [IO.File]::WriteAllText($teacherCorpusPath, "")
}

$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
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
        [string]$Text
    )

    $inputText = $Text + [Environment]::NewLine + "exit" + [Environment]::NewLine
    $lines = @($inputText | & $exe --load $LoadedCheckpoint --max-tokens 20 --temperature 0.15 --top-k 5 --no-online-learning 2>&1)

    $answerLine = $lines |
        ForEach-Object { [string]$_ } |
        Where-Object { $_ -match "ULTRON:s*(.*)$" } |
        Select-Object -First 1

    $answer = if ($answerLine -and $answerLine -match "ULTRON:s*(.*)$") {
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

function Train-UltronReplay {
    param(
        [string]$LoadedCheckpoint,
        [string]$TrainingCorpus,
        [string]$TargetCheckpoint,
        [int]$LessonCount
    )

    Write-Host "Training ULTRON on base corpus + $LessonCount accumulated lesson(s)..."

    $inputText = "exit" + [Environment]::NewLine
    $lines = @($inputText | & $exe --load $LoadedCheckpoint --train $TrainingCorpus --epochs $ReplayEpochs --lr $LearningRate --save $TargetCheckpoint --no-online-learning 2>&1)

    [PSCustomObject]@{
        Output = $lines
        ExitCode = $LASTEXITCODE
    }
}

function Write-CombinedCorpus {
    $base = [IO.File]::ReadAllText($baseCorpusPath)
    $lessons = [IO.File]::ReadAllText($teacherCorpusPath)
    $combined = $base.TrimEnd() + [Environment]::NewLine + [Environment]::NewLine + $lessons.TrimStart()
    [IO.File]::WriteAllText($combinedCorpusPath, $combined)
}

function Get-ExistingQuestions {
    if (!(Test-Path $teacherCorpusPath)) {
        return @()
    }

    return @(
        Get-Content $teacherCorpusPath |
            Where-Object { $_ -like "USER:*" } |
            ForEach-Object { $_.Substring(5).Trim() }
    )
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
    $existingQuestions = @(Get-ExistingQuestions)
    $questionSet = @{}

    foreach ($existingQuestion in $existingQuestions) {
        $questionSet[$existingQuestion.ToLowerInvariant()] = $true
    }

    Write-Host ""
    Write-Host "========================================"
    Write-Host " ULTRON <-> OLLAMA TEACHER SESSION"
    Write-Host "========================================"
    Write-Host "Teacher       : $Model"
    Write-Host "Rounds        : $Rounds"
    Write-Host "Replay epochs : $ReplayEpochs"
    Write-Host "Learning rate : $LearningRate"
    Write-Host "Replay corpus : $teacherCorpusPath"
    Write-Host ""

    for ($round = 1; $round -le $Rounds; $round++) {
        $recentQuestions = @(
            $existingQuestions |
                Select-Object -Last 25
        )

        $avoidText = if ($recentQuestions.Count -gt 0) {
            "Do not repeat these previous questions: " +
                ($recentQuestions -join " | ")
        } else {
            "There are no previous questions yet."
        }

        $prompt = @"
You are the teacher for a small language model called ULTRON.
Create exactly one useful, self-contained lesson.

Return JSON only with these fields:
{
  "question": "one clear question",
  "answer": "the correct answer in one or two sentences",
  "concept": "a short concept name"
}

Teach factual knowledge, reasoning, language, science, mathematics,
computer science, geography, history, or everyday useful knowledge.
Choose a topic different from recent lessons.
$avoidText

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
                temperature = 0.8
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

        $questionKey = $question.ToLowerInvariant()

        if ($questionSet.ContainsKey($questionKey)) {
            Write-Host "Teacher repeated an existing question; requesting a replacement..."

            $retryPrompt = @"
Create exactly one new useful lesson for ULTRON as JSON.
Return only "question", "answer", and "concept".
Do not repeat this question:
$question
Use a different topic and keep the answer to one or two sentences.
"@

            $retryBody = @{
                model = $Model
                prompt = $retryPrompt
                stream = $false
                think = $false
                format = "json"
                keep_alive = "5m"
                options = @{
                    temperature = 0.9
                }
            } | ConvertTo-Json -Depth 8

            $retryResponse = Invoke-RestMethod -Uri $api -Method Post -ContentType "application/json" -Body $retryBody -TimeoutSec 300

            if (![string]::IsNullOrWhiteSpace($retryResponse.response)) {
                $retryLesson = $retryResponse.response | ConvertFrom-Json

                if (![string]::IsNullOrWhiteSpace($retryLesson.question) -and
                    ![string]::IsNullOrWhiteSpace($retryLesson.answer)) {
                    $question = $retryLesson.question.Trim()
                    $answer = $retryLesson.answer.Trim()
                    $concept = if ($retryLesson.concept) { $retryLesson.concept.Trim() } else { "lesson" }
                }
            }
        }

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

        Add-Content -Path $teacherCorpusPath -Value ("USER: " + $question)
        Add-Content -Path $teacherCorpusPath -Value ("ULTRON: " + $answer)
        Add-Content -Path $teacherCorpusPath -Value ""

        $existingQuestions += $question
        $questionSet[$question.ToLowerInvariant()] = $true

        Write-CombinedCorpus

        Write-Host ""
        Write-Host "[ULTRON / REPLAY TRAINING]"
        $trained = Train-UltronReplay -LoadedCheckpoint $currentCheckpoint -TrainingCorpus $combinedCorpusPath -TargetCheckpoint $liveCheckpoint -LessonCount $existingQuestions.Count

        $lossLines = @(
            $trained.Output |
                ForEach-Object { [string]$_ } |
                Where-Object { $_ -match "^epoch " -or $_ -match "^Training complete" }
        )

        foreach ($line in $lossLines) {
            Write-Host $line
        }

        if ($trained.ExitCode -ne 0) {
            throw "ULTRON replay training failed with exit code $($trained.ExitCode)."
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
    Write-Host "========================================"
    Write-Host "Teacher session complete."
    Write-Host "Accumulated lessons: $($existingQuestions.Count)"
    Write-Host "Learned checkpoint : models/ultron_live.bin"
    Write-Host "Replay corpus      : $teacherCorpusPath"
    Write-Host "========================================"
} finally {
    if ($startedOllama -and $ollamaProcess) {
        try {
            $null = & $ollamaExe stop $Model 2>&1
        } catch {
            # The model may already be unloaded; cleanup should not mask
            # a successful teacher session.
        }

        Stop-Process -Id $ollamaProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped the Ollama daemon started by ULTRON."
    }
}
