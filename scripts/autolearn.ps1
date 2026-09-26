param(
    [string]$Checkpoint = "models/ultron_autolearn.bin",
    [string]$SeedData = "data/train.txt",
    [int]$MaxRounds = 100,
    [int]$BatchSentences = 5000,
    [int]$SourceSentences = 50000,
    [int]$BaseReplayChars = 100000,
    [int]$OnlineReplayChars = 60000,
    [int]$SeedEpochs = 1,
    [double]$LearningRate = 0.001,
    [int]$Speed = 1,
    [int]$QualityThreshold = 75,
    [int]$Patience = 3,
    [int]$TestEvery = 5,
    [int]$RefreshInternetEvery = 10,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build/Release/ultron.exe"
if (!(Test-Path $exe)) { $exe = Join-Path $repoRoot "build/ultron.exe" }
if (!(Test-Path $exe)) { throw "ULTRON executable not found. Build it first with .\scripts\build.ps1" }

if ($Speed -lt 1 -or $Speed -gt 10) { throw "Speed must be between 1 and 10." }
if ($QualityThreshold -lt 1 -or $QualityThreshold -gt 100) { throw "QualityThreshold must be between 1 and 100." }
if ($Patience -lt 1) { throw "Patience must be at least 1." }
if ($TestEvery -lt 1) { throw "TestEvery must be at least 1." }

$seedPath = Join-Path $repoRoot $SeedData
$checkpointPath = Join-Path $repoRoot $Checkpoint
$sourcePath = Join-Path $repoRoot "data/external/autolearn_tatoeba.txt"
$roundCorpusPath = Join-Path $repoRoot "data/external/autolearn_round.txt"
$savedCheckpointPath = Join-Path $repoRoot "models/ultron_autolearn_saved.bin"

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data/external") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "models") | Out-Null

if (!(Test-Path $seedPath)) { throw "Seed corpus not found: $SeedData" }

Write-Host "Running ULTRON smoke tests..."
& $exe --self-test
if ($LASTEXITCODE -ne 0) { throw "ULTRON smoke tests failed. Autonomous training was not started." }

$seedLines = @(Get-Content -Path $seedPath)
if ($seedLines.Count -lt 20) { throw "Seed corpus is too small: $SeedData" }

function Write-Utf8TextFile {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Get-ReplaySample {
    param([string[]]$Lines, [int]$MaxChars, [int]$BlockSize = 32)
    if ($Lines.Count -eq 0 -or $MaxChars -le 0) { return "" }

    $builder = [System.Text.StringBuilder]::new()
    $tries = [Math]::Max(12, [Math]::Ceiling($MaxChars / 1500))

    for ($attempt = 0; $attempt -lt $tries; $attempt++) {
        $start = if ($Lines.Count -le $BlockSize) { 0 } else { Get-Random -Minimum 0 -Maximum ($Lines.Count - $BlockSize) }

        for ($i = 0; $i -lt $BlockSize; $i++) {
            $line = $Lines[$start + $i]
            if ([string]::IsNullOrWhiteSpace($line)) { continue }

            $candidateLength = $line.Length + 1
            if (($builder.Length + $candidateLength) -gt $MaxChars) { return $builder.ToString() }

            [void]$builder.AppendLine($line)
        }

        if ($builder.Length -ge $MaxChars) { break }
    }

    return $builder.ToString()
}

function Get-OnlineBatch {
    param([string[]]$Lines, [int]$Count)
    if ($Lines.Count -eq 0) { throw "Internet corpus is empty." }

    $safeCount = [Math]::Min($Count, $Lines.Count)
    if ($safeCount -ge $Lines.Count) { return ($Lines -join [Environment]::NewLine) }

    $start = Get-Random -Minimum 0 -Maximum ($Lines.Count - $safeCount + 1)
    return ($Lines[$start..($start + $safeCount - 1)] -join [Environment]::NewLine)
}

function Download-InternetCorpus {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Connecting to the internet: Tatoeba English CC0 corpus"
    Write-Host " Source sentences: $SourceSentences"
    Write-Host "============================================================"

    & (Join-Path $repoRoot "scripts/download_tatoeba.ps1") -MaxSentences $SourceSentences

    $downloaded = Join-Path $repoRoot "data/external/tatoeba_en_cc0.txt"
    if (!(Test-Path $downloaded)) { throw "Tatoeba download did not create $downloaded" }

    Copy-Item $downloaded $sourcePath -Force
    Write-Host "Internet corpus ready: $sourcePath"
}

function Invoke-UltronPrompt {
    param([string]$LoadedCheckpoint, [string]$Prompt)

    $inputText = $Prompt + [Environment]::NewLine + "exit" + [Environment]::NewLine
    $lines = @(
        $inputText |
            & $exe --load $LoadedCheckpoint --max-tokens 24 --temperature 0.35 --top-k 5 --no-online-learning 2>&1 |
            ForEach-Object { [string]$_ }
    )

    $answerLine = $lines | Where-Object { $_ -match "^ULTRON:\s*(.*)$" } | Select-Object -First 1
    $answer = if ($answerLine -and $answerLine -match "^ULTRON:\s*(.*)$") { $matches[1].Trim() } else { "" }

    [PSCustomObject]@{ Answer = $answer; ExitCode = $LASTEXITCODE }
}

function Test-BasicEnglish {
    param([string]$LoadedCheckpoint)

    $tests = @(
        [PSCustomObject]@{ Prompt = "What does a computer do?"; Keywords = @("process","information","instruction","data") }
        [PSCustomObject]@{ Prompt = "What is a variable in programming?"; Keywords = @("value","store","data","change") }
        [PSCustomObject]@{ Prompt = "What does a router do?"; Keywords = @("network","data","device","internet") }
        [PSCustomObject]@{ Prompt = "What is water made of?"; Keywords = @("hydrogen","oxygen","h2o") }
        [PSCustomObject]@{ Prompt = "Why does rain fall from clouds?"; Keywords = @("water","cloud","gravity","drop","rain") }
        [PSCustomObject]@{ Prompt = "Explain what a dog is in one simple sentence."; Keywords = @("animal","dog","pet") }
        [PSCustomObject]@{ Prompt = "Give one sentence about a school."; Keywords = @("school","student","teacher","class") }
        [PSCustomObject]@{ Prompt = "Finish this sentence naturally: The boy opened the door and"; Keywords = @() }
    )

    $passed = 0
    $index = 0

    foreach ($test in $tests) {
        $index++
        $result = Invoke-UltronPrompt -LoadedCheckpoint $LoadedCheckpoint -Prompt $test.Prompt
        $answer = $result.Answer
        $lower = $answer.ToLowerInvariant()
        $ok = $true

        if ($result.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($answer)) { $ok = $false }

        $letters = ([regex]::Matches($answer, "[A-Za-z]")).Count
        $words = @($answer -split "\s+" | Where-Object { $_.Length -gt 0 })

        if ($letters -lt 8 -or $words.Count -lt 2 -or $words.Count -gt 35) { $ok = $false }

        $nonSpace = [Math]::Max(1, $answer.Length - ([regex]::Matches($answer, "\s")).Count)
        if (($letters / $nonSpace) -lt 0.45) { $ok = $false }

        if ($answer -match "<unk>|<pad>|<bos>|<eos>") { $ok = $false }
        if ($answer -match "([!?.,;:])\1\1" -or $answer -match "(.)\1\1\1\1") { $ok = $false }

        if ($words.Count -ge 6) {
            $repeatedNgram = $false
            $normalizedWords = @($words | ForEach-Object { $_.ToLowerInvariant() })

            for ($i = 0; $i -le $normalizedWords.Count - 3; $i++) {
                $ngram = "$($normalizedWords[$i]) $($normalizedWords[$i + 1]) $($normalizedWords[$i + 2])"
                for ($j = $i + 3; $j -le $normalizedWords.Count - 3; $j++) {
                    $ngram2 = "$($normalizedWords[$j]) $($normalizedWords[$j + 1]) $($normalizedWords[$j + 2])"
                    if ($ngram -eq $ngram2) { $repeatedNgram = $true; break }
                }
                if ($repeatedNgram) { break }
            }

            if ($repeatedNgram) { $ok = $false }
        }

        if ($test.Keywords.Count -gt 0) {
            $keywordHit = $false
            foreach ($keyword in $test.Keywords) {
                if ($lower.Contains($keyword)) { $keywordHit = $true; break }
            }
            if (!$keywordHit) { $ok = $false }
        }

        if ($ok) { $passed++; $status = "PASS" } else { $status = "FAIL" }
        Write-Host ("[{0}/8] {1} :: {2}" -f $index, $status, $answer)
    }

    [PSCustomObject]@{
        Score = [int][Math]::Round(($passed / $tests.Count) * 100.0)
        Passed = $passed
        Total = $tests.Count
    }
}

function Start-TrainingBatch {
    param([string]$LoadedCheckpoint, [string]$CorpusPath, [string]$TargetCheckpoint)

    $arguments = @(
        "--load", $LoadedCheckpoint,
        "--train", $CorpusPath,
        "--epochs", 1,
        "--lr", $LearningRate,
        "--save", $TargetCheckpoint,
        "--save-every", 1,
        "--speed", $Speed,
        "--non-interactive"
    )

    $process = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru
    $requestedSave = $false
    $requestedTest = $false
    $requestedStop = $false
    $pauseAfterBatch = $false

    while (!$process.HasExited) {
        Start-Sleep -Milliseconds 250

        while ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)

            switch ($key.Key) {
                "S" {
                    $requestedSave = $true
                    Write-Host "[HOTKEY] Save requested; saving at the next safe checkpoint."
                }
                "T" {
                    $requestedTest = $true
                    Write-Host "[HOTKEY] Test requested; testing after this batch."
                }
                "P" {
                    $pauseAfterBatch = !$pauseAfterBatch
                    if ($pauseAfterBatch) {
                        Write-Host "[HOTKEY] Pause requested after the current batch."
                    } else {
                        Write-Host "[HOTKEY] Pause cancelled."
                    }
                }
                "X" {
                    $requestedStop = $true
                    Write-Host "[HOTKEY] Stop requested; finishing the current batch safely."
                }
            }
        }
    }

    $process.Refresh()
    if ($process.ExitCode -ne 0) { throw "Training batch failed with exit code $($process.ExitCode)." }
    if (!(Test-Path $TargetCheckpoint)) { throw "Training completed but checkpoint was not created: $TargetCheckpoint" }

    if ($requestedSave) {
        Copy-Item $TargetCheckpoint $savedCheckpointPath -Force
        Write-Host "[SAVE] Safe checkpoint copied to $savedCheckpointPath"
    }

    [PSCustomObject]@{
        RequestedSave = $requestedSave
        RequestedTest = $requestedTest
        RequestedStop = $requestedStop
        PauseAfterBatch = $pauseAfterBatch
    }
}

if ($Fresh) {
    Remove-Item $checkpointPath -Force -ErrorAction SilentlyContinue
    Remove-Item $savedCheckpointPath -Force -ErrorAction SilentlyContinue
}

if (!(Test-Path $checkpointPath)) {
    Write-Host ""
    Write-Host "No autonomous checkpoint exists. Seeding from $SeedData"

    $seedArguments = @(
        "--train", $SeedData,
        "--epochs", $SeedEpochs,
        "--lr", $LearningRate,
        "--save", $Checkpoint,
        "--save-every", 1,
        "--speed", $Speed,
        "--non-interactive"
    )

    & $exe @seedArguments
    if ($LASTEXITCODE -ne 0) { throw "Initial seed training failed with exit code $LASTEXITCODE." }
    if (!(Test-Path $checkpointPath)) { throw "Initial seed training did not create $Checkpoint." }
}

Download-InternetCorpus
$internetLines = @(Get-Content -Path $sourcePath)

$currentCheckpoint = $checkpointPath
$bestScore = -1
$passingStreak = 0
$paused = $false
$stopRequested = $false
$testRequested = $false

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON AUTONOMOUS LEARNING MODE"
Write-Host "============================================================"
Write-Host "Core replay     : $SeedData ($BaseReplayChars chars/round)"
Write-Host "Internet replay : Tatoeba CC0 (~$OnlineReplayChars chars/round)"
Write-Host "Rounds          : $MaxRounds max"
Write-Host "Quality gate    : $QualityThreshold / 100"
Write-Host "Patience        : $Patience consecutive passing tests"
Write-Host "Auto-test every : $TestEvery rounds"
Write-Host ""
Write-Host "S = save | T = test | P = pause/resume | X = stop safely"
Write-Host "ULTRON never trains on its own generated answers."
Write-Host "============================================================"

for ($round = 1; $round -le $MaxRounds; $round++) {
    if ($paused) {
        Write-Host ""
        Write-Host "[PAUSED] Press P to resume or X to stop."

        while ($paused -and !$stopRequested) {
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                switch ($key.Key) {
                    "P" { $paused = $false; Write-Host "[HOTKEY] Resuming." }
                    "X" { $stopRequested = $true; Write-Host "[HOTKEY] Stopping." }
                    "T" { $testRequested = $true; Write-Host "[HOTKEY] Test queued." }
                    "S" {
                        if (Test-Path $currentCheckpoint) {
                            Copy-Item $currentCheckpoint $savedCheckpointPath -Force
                            Write-Host "[SAVE] Safe checkpoint copied to $savedCheckpointPath"
                        }
                    }
                }
            } else {
                Start-Sleep -Milliseconds 250
            }
        }
    }

    if ($stopRequested) { break }

    if ($round -gt 1 -and $RefreshInternetEvery -gt 0 -and (($round - 1) % $RefreshInternetEvery -eq 0)) {
        Download-InternetCorpus
        $internetLines = @(Get-Content -Path $sourcePath)
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " AUTONOMOUS ROUND $round / $MaxRounds"
    Write-Host "============================================================"

    $coreReplay = Get-ReplaySample -Lines $seedLines -MaxChars $BaseReplayChars
    $onlineCount = [Math]::Min($BatchSentences, [Math]::Max(1, [Math]::Floor($OnlineReplayChars / 80)))
    $internetBatch = Get-OnlineBatch -Lines $internetLines -Count $onlineCount

    $combined = $coreReplay.TrimEnd() + [Environment]::NewLine + [Environment]::NewLine + $internetBatch.Trim()
    Write-Utf8TextFile -Path $roundCorpusPath -Text $combined

    Write-Host "[LEARN] Training on core replay + internet text."
    $batchResult = Start-TrainingBatch -LoadedCheckpoint $currentCheckpoint -CorpusPath $roundCorpusPath -TargetCheckpoint $currentCheckpoint

    if ($batchResult.PauseAfterBatch) { $paused = $true }
    if ($batchResult.RequestedTest) { $testRequested = $true }

    if (($round % $TestEvery -eq 0) -or $testRequested -or $round -eq 1) {
        $testRequested = $false

        Write-Host ""
        Write-Host "[SELF-CHECK] Testing basic English..."
        $quality = Test-BasicEnglish -LoadedCheckpoint $currentCheckpoint

        Write-Host ""
        Write-Host ("[SELF-CHECK] Score: {0}/100 ({1}/{2} tests passed)" -f $quality.Score, $quality.Passed, $quality.Total)

        if ($quality.Score -gt $bestScore) {
            $bestScore = $quality.Score
            $passingStreak = if ($quality.Score -ge $QualityThreshold) { 1 } else { 0 }
            Write-Host "[SELF-CHECK] New best score: $bestScore"
        } elseif ($quality.Score -ge $QualityThreshold) {
            $passingStreak++
            Write-Host "[SELF-CHECK] Passing streak: $passingStreak / $Patience"
        } else {
            $passingStreak = 0
        }

        if ($quality.Score -ge $QualityThreshold -and $passingStreak -ge $Patience) {
            Write-Host ""
            Write-Host "============================================================"
            Write-Host " ULTRON DECIDED THE BASIC-ENGLISH GATE IS SATISFIED"
            Write-Host " Score : $($quality.Score)/100"
            Write-Host "============================================================"

            Copy-Item $currentCheckpoint $savedCheckpointPath -Force
            Write-Host "Final safe checkpoint: $savedCheckpointPath"
            break
        }
    }

    if ($batchResult.RequestedStop) {
        Write-Host "[STOP] Safe stop requested."
        Copy-Item $currentCheckpoint $savedCheckpointPath -Force
        Write-Host "[STOP] Saved: $savedCheckpointPath"
        break
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON AUTONOMOUS LEARNING FINISHED"
Write-Host " Active checkpoint : $currentCheckpoint"
Write-Host " Best quality      : $bestScore/100"
Write-Host " Saved checkpoint  : $savedCheckpointPath"
Write-Host "============================================================"
