cmake_minimum_required(VERSION 3.20)

project(ULTRON_LLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ULTRON_SOURCES
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

add_executable(ultron ${ULTRON_SOURCES})
target_include_directories(ultron PRIVATE include)

if (MSVC)
    target_compile_options(ultron PRIVATE /W4)
    target_compile_options(ultron PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/fp:fast>)
else()
    target_compile_options(ultron PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ultron PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-march=native>)
endif()

include(CTest)
if (BUILD_TESTING)
    add_executable(ultron_smoke_tests
        tests/smoke.cpp
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
    target_include_directories(ultron_smoke_tests PRIVATE include)
    add_test(NAME ultron_smoke_tests COMMAND ultron_smoke_tests)
endif()
