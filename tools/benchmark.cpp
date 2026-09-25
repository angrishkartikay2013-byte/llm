#include "model.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main() {
    ULTRONModel model;
    constexpr int iterations = 20;

    const std::string prompt =
        "ultron is a";

    const auto start = std::chrono::steady_clock::now();
    std::size_t characters = 0;

    for (int i = 0; i < iterations; ++i) {
        const std::string output =
            model.generate(prompt, 16, 0.8f, 8,
                           static_cast<unsigned int>(i));
        characters += output.size();
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    const double seconds = elapsed.count();
    const double generations_per_second =
        seconds > 0.0 ? iterations / seconds : 0.0;

    std::cout
        << "ULTRON inference benchmark\n"
        << "Iterations: " << iterations << '\n'
        << "Elapsed seconds: " << seconds << '\n'
        << "Generations/sec: " << generations_per_second << '\n'
        << "Characters produced: " << characters << '\n';

    return 0;
}
