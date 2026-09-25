#include "metrics.hpp"

#include "trainer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

EvaluationMetrics evaluate_predictions(
    const std::vector<std::vector<float>>& logits,
    const std::vector<std::size_t>& targets) {

    if (logits.size() != targets.size()) {
        throw std::invalid_argument(
            "Logit and target counts differ");
    }

    EvaluationMetrics metrics;
    metrics.samples = logits.size();

    if (metrics.samples == 0) {
        return metrics;
    }

    double total_loss = 0.0;
    std::size_t correct = 0;

    for (std::size_t sample = 0;
         sample < logits.size();
         ++sample) {

        const auto probabilities =
            ultron_softmax(logits[sample]);

        const std::size_t target = targets[sample];

        if (target >= probabilities.size()) {
            throw std::out_of_range(
                "Evaluation target exceeds vocabulary");
        }

        total_loss += ultron_cross_entropy_loss(
            probabilities,
            target);

        const std::size_t prediction =
            static_cast<std::size_t>(
                std::max_element(
                    probabilities.begin(),
                    probabilities.end()) -
                probabilities.begin());

        if (prediction == target) {
            ++correct;
        }
    }

    metrics.mean_loss =
        total_loss /
        static_cast<double>(metrics.samples);

    metrics.perplexity =
        std::exp(metrics.mean_loss);

    metrics.accuracy =
        static_cast<double>(correct) /
        static_cast<double>(metrics.samples);

    return metrics;
}
