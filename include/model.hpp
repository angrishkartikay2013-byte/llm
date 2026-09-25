#pragma once

#include <cstddef>
#include <memory>
#include <string>

class ULTRONModel {
public:
    ULTRONModel();
    ~ULTRONModel();

    ULTRONModel(const ULTRONModel&) = delete;
    ULTRONModel& operator=(const ULTRONModel&) = delete;
    ULTRONModel(ULTRONModel&&) noexcept;
    ULTRONModel& operator=(ULTRONModel&&) noexcept;

    void train(
        const std::string& text,
        std::size_t epochs = 1,
        float learning_rate = 0.003f);

    std::string generate(
        const std::string& prompt,
        std::size_t max_new_tokens = 1,
        float temperature = 0.8f,
        std::size_t top_k = 8,
        unsigned int seed = 42) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
