param(
    [string]$Output = "data/external/tatoeba_en_cc0.txt",
    [int]$MaxSentences = 100000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoRoot $Output
$archivePath = Join-Path $repoRoot "data/external/tatoeba_en_cc0.tsv.bz2"
$rawPath = Join-Path $repoRoot "data/external/tatoeba_en_cc0.tsv"
$outputDirectory = Split-Path -Parent $outputPath

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$url = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2"

Write-Host "Downloading Tatoeba English CC0 sentences..."
Invoke-WebRequest -Uri $url -OutFile $archivePath

Write-Host "Extracting..."
& tar.exe -xjf $archivePath -O | Out-File -FilePath $rawPath -Encoding utf8

if (!(Test-Path $rawPath)) {
    throw "Tatoeba archive extraction failed."
}

Write-Host "Converting TSV to plain text..."

$count = 0
$writer = [System.IO.StreamWriter]::new($outputPath, $false, [System.Text.UTF8Encoding]::new($false))

try {
    foreach ($line in [System.IO.File]::ReadLines($rawPath)) {
        if ($count -ge $MaxSentences) {
            break
        }

        $fields = $line -split [char]9, 3

        if ($fields.Count -eq 3) {
            $text = $fields[2].Trim()

            if ($text.Length -gt 0) {
                $writer.WriteLine($text)
                $count++
            }
        }
    }
}
finally {
    $writer.Dispose()
}

Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
Remove-Item $rawPath -Force -ErrorAction SilentlyContinue

Write-Host "Wrote $count sentences to $outputPath"
Write-Host "These sentences are from Tatoeba's CC0 English export."