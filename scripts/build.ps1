$ErrorActionPreference = "Stop"

$buildDirectory = "build"
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
$ninjaCommand = Get-Command ninja -ErrorAction SilentlyContinue

if ($null -ne $cmakeCommand) {
    cmake -S . -B $buildDirectory -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }

    cmake --build $buildDirectory --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }
} elseif ($null -ne $ninjaCommand -and (Test-Path (Join-Path $buildDirectory "build.ninja"))) {
    Write-Host "CMake is not on PATH; using the existing Ninja build directory configuration."
    $jobs = [Math]::Max(1, [Environment]::ProcessorCount)
    ninja -C $buildDirectory -j $jobs
    if ($LASTEXITCODE -ne 0) { throw "Ninja build failed." }
} else {
    throw "CMake is not on PATH and no usable Ninja build was found. Install CMake or add it to PATH."
}

Write-Host ""
Write-Host "ULTRON build complete."
Write-Host "Run tests with: ctest --test-dir build -C Release --output-on-failure"
