#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

struct ModelEvaluation {
    double mean_loss = 0.0;
    double perplexity = 0.0;
    double accuracy = 0.0;
    std::size_t samples = 0;
};

class ULTRONModel {
public:
    ULTRONModel();
    ~ULTRONModel();

    ULTRONModel(const ULTRONModel&) = delete;
    ULTRONModel& operator=(const ULTRONModel&) = delete;
    ULTRONModel(ULTRONModel&&) noexcept;
    ULTRONModel& operator=(ULTRONModel&&) noexcept;

    float train(
        const std::string& text,
        std::size_t epochs = 1,
        float learning_rate = 0.001f,
        const std::function<void(
            std::size_t,
            float)>& progress = {},
        std::size_t speed = 1);

    ModelEvaluation evaluate(
        const std::string& text) const;

    std::string generate(
        const std::string& prompt,
        std::size_t max_new_tokens = 1,
        float temperature = 0.8f,
        std::size_t top_k = 8,
        unsigned int seed = 42) const;

    bool save_checkpoint(const std::string& path) const;
    bool load_checkpoint(const std::string& path);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
