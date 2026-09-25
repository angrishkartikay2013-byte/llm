$ErrorActionPreference = "Stop"

cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release

Write-Host ""
Write-Host "ULTRON build complete."
Write-Host "Run tests with: ctest --test-dir build -C Release --output-on-failure"