#pragma once

#include <cstddef>
#include <vector>

struct EvaluationMetrics {
    double mean_loss = 0.0;
    double perplexity = 0.0;
    double accuracy = 0.0;
    std::size_t samples = 0;
};

EvaluationMetrics evaluate_predictions(
    const std::vector<std::vector<float>>& logits,
    const std::vector<std::size_t>& targets);
