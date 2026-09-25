cmake_minimum_required(VERSION 3.20)

project(ULTRON_LLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ultron
    src/main.cpp
    src/config.cpp
    src/dataset.cpp
    src/metrics.cpp
    src/tokenizer.cpp
    src/tensor.cpp
    src/embedding.cpp
    src/attention.cpp
    src/multi_head_attention.cpp
    src/layer_norm.cpp
    src/transformer.cpp
    src/autograd.cpp
    src/optimizer.cpp
    src/trainer.cpp
    src/model.cpp
)

target_include_directories(ultron PRIVATE include)

if (MSVC)
    target_compile_options(ultron PRIVATE /W4)
    target_compile_options(ultron PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast>)
else()
    target_compile_options(ultron PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ultron PRIVATE $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-ffast-math> $<$<CONFIG:Release>:-march=native>)
endif()
