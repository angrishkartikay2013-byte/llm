# ULTRON LLM

ULTRON is a CPU-oriented Large Language Model engine built from scratch in C++20.

## Build

Use `cmake -S . -B build -DBUILD_TESTING=ON` and then `cmake --build build --config Release`.

Run tests with `ctest --test-dir build -C Release --output-on-failure`.

## Benchmark

Configure with `-DULTRON_BUILD_BENCHMARK=ON`, build, and run `ultron_benchmark`.

The benchmark reports generations per second for the local inference engine.