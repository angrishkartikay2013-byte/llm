#include "transformer.hpp"

#include <cmath>

namespace {
float gelu(float x) {
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    return 0.5f * x *
           (1.0f + std::tanh(
               kSqrtTwoOverPi * (x + 0.044715f * x * x * x)
           ));
}
}

TransformerBlock::TransformerBlock(int embedding_size)
    : embedding_size_(embedding_size) {}

std::vector<float> TransformerBlock::feed_forward(
    const std::vector<float>& input) const {

    std::vector<float> output = input;
    for (float& value : output) {
        value = gelu(value);
    }
    return output;
}

std::vector<std::vector<float>> TransformerBlock::forward(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& attention_weights) const {

    std::vector<std::vector<float>> output = embeddings;

    for (std::size_t i = 0; i < output.size(); ++i) {
        const float weight =
            i < attention_weights.size() ? attention_weights[i] : 1.0f;

        for (float& value : output[i]) {
            value *= weight;
        }

        output[i] = feed_forward(output[i]);

        if (static_cast<int>(output[i].size()) != embedding_size_) {
            output[i].resize(static_cast<std::size_t>(embedding_size_), 0.0f);
        }
    }

    return output;
}
