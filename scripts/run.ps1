param(
    [string]$Checkpoint = "models/ultron.bin",
    [int]$MaxTokens = 16,
    [double]$Temperature = 0.8,
    [int]$TopK = 8
)

$ErrorActionPreference = "Stop"

$exe = "build/Release/ultron.exe"
if (!(Test-Path $exe)) { $exe = "build/ultron.exe" }
if (!(Test-Path $exe)) { throw "ULTRON executable not found. Run scripts/build.ps1 first." }

& $exe --load $Checkpoint --max-tokens $MaxTokens --temperature $Temperature --top-k $TopK