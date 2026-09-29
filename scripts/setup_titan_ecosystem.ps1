# TITAN External Ecosystem Bootstrap
# Run from Windows PowerShell.
# External repositories are cloned to E:\Titan\repos, NOT vendored into the C++ LLM source tree.

[CmdletBinding()]
param(
    [switch]$SkipExisting,
    [switch]$ReferenceOnly
)

$ErrorActionPreference = "Stop"

$TitanRoot = "E:\Titan"
$ReposRoot = Join-Path $TitanRoot "repos"

$repos = @(
    @{ Name="UFO"; Url="https://github.com/microsoft/UFO.git"; Path="UFO"; Install=$true },
    @{ Name="A2A"; Url="https://github.com/a2aproject/A2A.git"; Path="A2A"; Install=$true },
    @{ Name="a2a-python"; Url="https://github.com/a2aproject/a2a-python.git"; Path="a2a-python"; Install=$true },
    @{ Name="LangGraph"; Url="https://github.com/langchain-ai/langgraph.git"; Path="LangGraph"; Install=$true },
    @{ Name="mcp-servers"; Url="https://github.com/modelcontextprotocol/servers.git"; Path="mcp-servers"; Install=$true },
    @{ Name="ComfyUI"; Url="https://github.com/comfyanonymous/ComfyUI.git"; Path="ComfyUI"; Install=$true },
    @{ Name="Wan2.1"; Url="https://github.com/Wan-Video/Wan2.1.git"; Path="Wan2.1"; Install=$true },
    @{ Name="browser-use"; Url="https://github.com/browser-use/browser-use.git"; Path="browser-use"; Install=$true },
    @{ Name="openagent"; Url="https://github.com/the-open-agent/openagent.git"; Path="openagent"; Install=$false },
    @{ Name="agent-memory"; Url="https://github.com/neo4j-labs/agent-memory.git"; Path="agent-memory"; Install=$false },
    @{ Name="mnem"; Url="https://github.com/Uranid/mnem.git"; Path="mnem"; Install=$true },
    @{ Name="PlugMem"; Url="https://github.com/TIMAN-group/PlugMem.git"; Path="PlugMem"; Install=$false }
)

New-Item -ItemType Directory -Force -Path $ReposRoot | Out-Null

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "        TITAN ECOSYSTEM BOOTSTRAP        " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Target: $ReposRoot"
Write-Host ""

foreach ($repo in $repos) {
    if ($ReferenceOnly -and $repo.Install) {
        continue
    }

    $target = Join-Path $ReposRoot $repo.Path

    if (Test-Path (Join-Path $target ".git")) {
        if ($SkipExisting) {
            Write-Host "[SKIP] $($repo.Name) already exists." -ForegroundColor Yellow
            continue
        }

        Write-Host "[UPDATE] $($repo.Name)" -ForegroundColor Cyan
        git -C $target pull --ff-only
        continue
    }

    if (Test-Path $target) {
        Write-Host "[ERROR] $target exists but is not a Git repository." -ForegroundColor Red
        continue
    }

    Write-Host "[CLONE] $($repo.Name)" -ForegroundColor Green
    git clone --depth 1 $repo.Url $target

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone $($repo.Name)."
    }
}

Write-Host ""
Write-Host "TITAN ecosystem bootstrap complete." -ForegroundColor Green
Write-Host "Heavy model dependencies are intentionally NOT installed automatically." -ForegroundColor Yellow
Write-Host "Each subsystem should get its own environment to avoid dependency conflicts." -ForegroundColor Yellow
