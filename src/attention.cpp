#include "attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Attention::Attention() = default;

std::vector<float> Attention::softmax(
    const std::vector<float>& scores) const {

    if (scores.empty()) {
        return {};
    }

    const float max_value =
        *std::max_element(scores.begin(), scores.end());

    std::vector<float> probabilities(scores.size());
    float sum = 0.0f;

    for (std::size_t i = 0; i < scores.size(); ++i) {
        probabilities[i] = std::exp(scores[i] - max_value);
        sum += probabilities[i];
    }

    if (sum <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("Softmax produced an invalid normalization");
    }

    for (float& probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

std::vector<std::vector<float>> Attention::forward(
    const std::vector<std::vector<float>>& embeddings) const {

    if (embeddings.empty()) {
        return {};
    }

    const std::size_t dimension = embeddings.front().size();
    if (dimension == 0) {
        throw std::invalid_argument("Attention requires non-empty embeddings");
    }

    for (const auto& row : embeddings) {
        if (row.size() != dimension) {
            throw std::invalid_argument("Attention received ragged embeddings");
        }
    }

    const std::size_t sequence_length = embeddings.size();
    std::vector<std::vector<float>> output(
        sequence_length,
        std::vector<float>(dimension, 0.0f));

    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));

    for (std::size_t query = 0; query < sequence_length; ++query) {
        std::vector<float> scores(sequence_length, 0.0f);

        for (std::size_t key = 0; key <= query; ++key) {
            float dot = 0.0f;
            for (std::size_t d = 0; d < dimension; ++d) {
                dot += embeddings[query][d] * embeddings[key][d];
            }
            scores[key] = dot * scale;
        }

        for (std::size_t key = query + 1; key < sequence_length; ++key) {
            scores[key] = -std::numeric_limits<float>::infinity();
        }

        const auto weights = softmax(scores);

        for (std::size_t key = 0; key <= query; ++key) {
            for (std::size_t d = 0; d < dimension; ++d) {
                output[query][d] += weights[key] * embeddings[key][d];
            }
        }
    }

    return output;
}
