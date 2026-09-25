# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

## Build and test

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The smoke tests cover tensor math, tokenization, optimization, training, evaluation, and generation.
