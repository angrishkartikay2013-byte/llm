param(
    [string]$TrainOutput = "data/external/ultrachat_train_sft.txt",
    [string]$ValidationOutput = "data/external/ultrachat_test_sft.txt",
    [int]$TrainConversations = 3000,
    [int]$ValidationConversations = 500,
    [int]$MaxConversationChars = 12000,
    [int]$RequestBatchSize = 100,
    [int]$StartTrainOffset = 0,
    [int]$StartValidationOffset = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainPath = Join-Path $repoRoot $TrainOutput
$validationPath = Join-Path $repoRoot $ValidationOutput
$manifestPath = Join-Path $repoRoot "data/external/ultrachat_manifest.txt"

$dataset = "HuggingFaceH4/ultrachat_200k"
$config = "default"

if ($RequestBatchSize -lt 1 -or $RequestBatchSize -gt 100) { throw "RequestBatchSize must be between 1 and 100." }
if ($TrainConversations -lt 1) { throw "TrainConversations must be at least 1." }
if ($ValidationConversations -lt 1) { throw "ValidationConversations must be at least 1." }
if ($MaxConversationChars -lt 500) { throw "MaxConversationChars is too small." }

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $trainPath) | Out-Null

function Normalize-Message {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return "" }
    $clean = [regex]::Replace($Text, "\r\n|\r", "\n")
    $clean = $clean.Trim()
    $clean = [regex]::Replace($clean, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "")
    return $clean.Trim()
}

function Get-StringHash {
    param([string]$Text)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-Conversations {
    param(
        [string]$Split,
        [int]$Wanted,
        [int]$StartOffset,
        [int]$MaxChars,
        [string]$OutputPath
    )

    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $writer = [System.IO.StreamWriter]::new($OutputPath, $false, [System.Text.UTF8Encoding]::new($false))
    $accepted = 0
    $requestedOffset = $StartOffset
    $apiBase = "https://datasets-server.huggingface.co/rows"

    try {
        while ($accepted -lt $Wanted) {
            $remaining = $Wanted - $accepted
            $length = [Math]::Min($RequestBatchSize, [Math]::Max(1, $remaining))
            $uri = $apiBase + "?dataset=" + [Uri]::EscapeDataString($dataset) + "&config=" + [Uri]::EscapeDataString($config) + "&split=" + [Uri]::EscapeDataString($Split) + "&offset=" + $requestedOffset + "&length=" + $length

            Write-Host ("Fetching {0}: offset {1}, {2} rows..." -f $Split, $requestedOffset, $length)

            $response = $null
            $attempt = 0
            while ($null -eq $response) {
                $attempt++
                try {
                    $response = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 120
                } catch {
                    if ($attempt -ge 5) {
                        throw "Hugging Face request failed after $attempt attempts: $($_.Exception.Message)"
                    }
                    $delay = [Math]::Min(30, 2 * $attempt)
                    Write-Warning "Request failed; retrying in $delay seconds..."
                    Start-Sleep -Seconds $delay
                }
            }

            if ($null -eq $response.rows -or $response.rows.Count -eq 0) {
                throw "No more rows were returned for split $Split at offset $requestedOffset."
            }

            foreach ($item in $response.rows) {
                if ($accepted -ge $Wanted) { break }

                $messages = @($item.row.messages)
                if ($messages.Count -lt 2) { continue }

                $parts = [System.Collections.Generic.List[string]]::new()
                $seenUser = $false
                $seenAssistant = $false

                foreach ($message in $messages) {
                    $role = [string]$message.role
                    $content = Normalize-Message ([string]$message.content)
                    if ([string]::IsNullOrWhiteSpace($content)) { continue }

                    if ($role -eq "user") {
                        [void]$parts.Add("USER: " + $content)
                        $seenUser = $true
                    } elseif ($role -eq "assistant") {
                        if (!$seenUser) { continue }
                        [void]$parts.Add("ULTRON: " + $content)
                        $seenAssistant = $true
                    }
                }

                if (!$seenUser -or !$seenAssistant) { continue }

                while ($parts.Count -gt 0 -and $parts[$parts.Count - 1] -like "USER:*") {
                    $parts.RemoveAt($parts.Count - 1)
                }

                if ($parts.Count -lt 2) { continue }

                $conversation = ($parts -join [Environment]::NewLine).Trim()
                if ($conversation.Length -lt 100 -or $conversation.Length -gt $MaxChars) { continue }

                if ($conversation -match "<\|[^>]{1,80}\|>" -or $conversation -match "([!?.,;:])\1\1" -or $conversation -match "(.)\1\1\1\1") {
                    continue
                }

                $hash = Get-StringHash $conversation
                if (!$seen.Add($hash)) { continue }

                $writer.WriteLine($conversation)
                $writer.WriteLine()
                $accepted++

                if (($accepted % 100) -eq 0) {
                    Write-Host ("Accepted {0}/{1} conversations from {2}." -f $accepted, $Wanted, $Split)
                }
            }

            $requestedOffset += [int]$response.rows.Count

            if ($requestedOffset -ge [int]$response.num_rows_total -and $accepted -lt $Wanted) {
                throw "Reached the end of split $Split before collecting $Wanted usable conversations. Accepted: $accepted."
            }
        }
    } finally {
        $writer.Dispose()
    }

    return $accepted
}

Write-Host ""
Write-Host "============================================================"
Write-Host " ULTRON Hugging Face control corpus"
Write-Host " Dataset : $dataset"
Write-Host " Train   : train_sft"
Write-Host " Test    : test_sft"
Write-Host "============================================================"
Write-Host ""

$trainCount = Get-Conversations -Split "train_sft" -Wanted $TrainConversations -StartOffset $StartTrainOffset -MaxChars $MaxConversationChars -OutputPath $trainPath
$validationCount = Get-Conversations -Split "test_sft" -Wanted $ValidationConversations -StartOffset $StartValidationOffset -MaxChars $MaxConversationChars -OutputPath $validationPath

$trainBytes = (Get-Item $trainPath).Length
$validationBytes = (Get-Item $validationPath).Length

$manifest = @(
    "dataset=$dataset"
    "config=$config"
    "train_split=train_sft"
    "validation_split=test_sft"
    "train_conversations=$trainCount"
    "validation_conversations=$validationCount"
    "train_bytes=$trainBytes"
    "validation_bytes=$validationBytes"
    "max_conversation_chars=$MaxConversationChars"
    "downloaded_utc=$([DateTime]::UtcNow.ToString("o"))"
)

[System.IO.File]::WriteAllLines($manifestPath, $manifest, [System.Text.UTF8Encoding]::new($false))

Write-Host ""
Write-Host "============================================================"
Write-Host " HF control corpus ready"
Write-Host " Train      : $trainPath ($trainBytes bytes)"
Write-Host " Validation : $validationPath ($validationBytes bytes)"
Write-Host " Manifest   : $manifestPath"
Write-Host "============================================================"
