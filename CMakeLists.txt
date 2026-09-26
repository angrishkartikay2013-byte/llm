cmake_minimum_required(VERSION 3.20)

project(ULTRON_LLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ULTRON_CORE_SOURCES
    src/config.cpp
    src/conversation.cpp
    src/dictionary.cpp
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
    src/self_test.cpp
)

add_executable(ultron
    src/main.cpp
    ${ULTRON_CORE_SOURCES}
)
target_include_directories(ultron PRIVATE include)

include(CheckIPOSupported)
check_ipo_supported(
    RESULT ULTRON_IPO_SUPPORTED
    OUTPUT ULTRON_IPO_ERROR)
if (ULTRON_IPO_SUPPORTED)
    set_property(
        TARGET ultron
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

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
    add_test(NAME ultron_smoke_tests COMMAND ultron --self-test)
endif()

option(ULTRON_BUILD_BENCHMARK "Build the CPU inference benchmark" OFF)
if (ULTRON_BUILD_BENCHMARK)
    add_executable(ultron_benchmark
        tools/benchmark.cpp
        ${ULTRON_CORE_SOURCES}
    )
    target_include_directories(ultron_benchmark PRIVATE include)
endif()
