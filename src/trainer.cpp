#include "trainer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

std::vector<float> ultron_softmax(
    const std::vector<float>& logits) {

    if (logits.empty()) {
        return {};
    }

    const float max_logit =
        *std::max_element(logits.begin(), logits.end());

    std::vector<float> probabilities(logits.size());
    float total = 0.0f;

    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] =
            std::exp(logits[i] - max_logit);
        total += probabilities[i];
    }

    if (total <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("Softmax normalization failed");
    }

    for (float& probability : probabilities) {
        probability /= total;
    }

    return probabilities;
}

float ultron_cross_entropy_loss(
    const std::vector<float>& probabilities,
    std::size_t target) {

    if (target >= probabilities.size()) {
        throw std::out_of_range("Training target is outside vocabulary");
    }

    constexpr float epsilon = 1e-8f;

    return -std::log(
        std::max(probabilities[target], epsilon));
}

void ultron_output_gradient(
    const std::vector<float>& hidden,
    const std::vector<float>& probabilities,
    std::size_t target,
    std::vector<float>& gradient) {

    if (target >= probabilities.size()) {
        throw std::out_of_range("Training target is outside vocabulary");
    }

    gradient.assign(
        probabilities.size() * hidden.size(),
        0.0f);

    for (std::size_t token = 0;
         token < probabilities.size();
         ++token) {

        const float error =
            probabilities[token] -
            (token == target ? 1.0f : 0.0f);

        for (std::size_t dimension = 0;
             dimension < hidden.size();
             ++dimension) {

            gradient[
                token * hidden.size() + dimension
            ] = error * hidden[dimension];
        }
    }
}
