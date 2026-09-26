#include "transformer.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <istream>
#include <ostream>

TransformerBlock::Matrix TransformerBlock::random_matrix(
    std::size_t rows,
    std::size_t cols,
    unsigned int seed) {

    Matrix matrix(rows, std::vector<float>(cols, 0.0f));
    std::mt19937 generator(seed);

    const float limit = std::sqrt(
        6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    for (auto& row : matrix) {
        for (float& value : row) {
            value = distribution(generator);
        }
    }

    return matrix;
}

TransformerBlock::TransformerBlock(
    std::size_t embedding_size,
    std::size_t num_heads,
    std::size_t feed_forward_size)
    : embedding_size_(embedding_size),
      num_heads_(num_heads),
      head_size_(embedding_size / num_heads),
      feed_forward_size_(feed_forward_size),
      query_weight_(random_matrix(embedding_size, embedding_size, 11)),
      key_weight_(random_matrix(embedding_size, embedding_size, 12)),
      value_weight_(random_matrix(embedding_size, embedding_size, 13)),
      output_weight_(random_matrix(embedding_size, embedding_size, 14)),
      feed_forward_in_(random_matrix(embedding_size, feed_forward_size, 15)),
      feed_forward_out_(random_matrix(feed_forward_size, embedding_size, 16)) {

    if (embedding_size_ == 0 || num_heads_ == 0 ||
        embedding_size_ % num_heads_ != 0 ||
        feed_forward_size_ == 0) {
        throw std::invalid_argument("Invalid Transformer configuration");
    }
}

std::vector<float> TransformerBlock::linear(
    const std::vector<float>& input,
    const Matrix& weights) {

    if (input.size() != weights.size()) {
        throw std::invalid_argument("Linear input shape mismatch");
    }

    std::vector<float> output(
        weights.front().size(), 0.0f);

    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights[i].size(); ++j) {
            output[j] += input[i] * weights[i][j];
        }
    }

    return output;
}

std::vector<float> TransformerBlock::layer_norm(
    const std::vector<float>& input) {

    if (input.empty()) {
        return {};
    }

    float mean = 0.0f;
    for (float value : input) {
        mean += value;
    }
    mean /= static_cast<float>(input.size());

    float variance = 0.0f;
    for (float value : input) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<float>(input.size());

    constexpr float epsilon = 1e-5f;
    const float inverse_std =
        1.0f / std::sqrt(variance + epsilon);

    std::vector<float> normalized(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        normalized[i] = (input[i] - mean) * inverse_std;
    }

    return normalized;
}

float TransformerBlock::gelu(float value) {
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    return 0.5f * value *
        (1.0f + std::tanh(
            kSqrtTwoOverPi *
            (value + 0.044715f * value * value * value)));
}

std::vector<std::vector<float>> TransformerBlock::forward(
    const std::vector<std::vector<float>>& embeddings) const {

    if (embeddings.empty()) {
        return {};
    }

    for (const auto& row : embeddings) {
        if (row.size() != embedding_size_) {
            throw std::invalid_argument(
                "Transformer embedding dimension mismatch");
        }
    }

    const std::size_t sequence_length = embeddings.size();

    std::vector<std::vector<float>> queries(sequence_length);
    std::vector<std::vector<float>> keys(sequence_length);
    std::vector<std::vector<float>> values(sequence_length);

    for (std::size_t i = 0; i < sequence_length; ++i) {
        queries[i] = linear(embeddings[i], query_weight_);
        keys[i] = linear(embeddings[i], key_weight_);
        values[i] = linear(embeddings[i], value_weight_);
    }

    std::vector<std::vector<float>> attention_output(
        sequence_length,
        std::vector<float>(embedding_size_, 0.0f));

    for (std::size_t query = 0; query < sequence_length; ++query) {
        for (std::size_t head = 0; head < num_heads_; ++head) {
            const std::size_t offset = head * head_size_;
            std::vector<float> scores(query + 1, 0.0f);

            float max_score = -std::numeric_limits<float>::infinity();

            for (std::size_t key = 0; key <= query; ++key) {
                float dot = 0.0f;

                for (std::size_t d = 0; d < head_size_; ++d) {
                    dot += queries[query][offset + d] *
                           keys[key][offset + d];
                }

                scores[key] =
                    dot / std::sqrt(static_cast<float>(head_size_));
                max_score = std::max(max_score, scores[key]);
            }

            float sum = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }

            for (std::size_t key = 0; key <= query; ++key) {
                const float weight = scores[key] / sum;

                for (std::size_t d = 0; d < head_size_; ++d) {
                    attention_output[query][offset + d] +=
                        weight * values[key][offset + d];
                }
            }
        }

        attention_output[query] =
            linear(attention_output[query], output_weight_);

        for (std::size_t d = 0; d < embedding_size_; ++d) {
            attention_output[query][d] += embeddings[query][d];
        }

        attention_output[query] =
            layer_norm(attention_output[query]);
    }

    std::vector<std::vector<float>> output = attention_output;

    for (std::size_t i = 0; i < sequence_length; ++i) {
        const auto hidden = linear(output[i], feed_forward_in_);
        std::vector<float> activated(hidden.size());

        for (std::size_t j = 0; j < hidden.size(); ++j) {
            activated[j] = gelu(hidden[j]);
        }

        const auto projected =
            linear(activated, feed_forward_out_);

        for (std::size_t d = 0; d < embedding_size_; ++d) {
            output[i][d] += projected[d];
        }

        output[i] = layer_norm(output[i]);
    }

    return output;
}

std::vector<std::vector<float>> TransformerBlock::forward(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& attention_weights) const {

    // The architecture now computes its own causal multi-head attention.
    // Keep the old signature source-compatible with the early engine.
    (void)attention_weights;
    return forward(embeddings);
}


bool TransformerBlock::save(
    std::ostream& output) const {

    if (!transformer_write_u64(
            output,
            static_cast<std::uint64_t>(embedding_size_)) ||
        !transformer_write_u64(
            output,
            static_cast<std::uint64_t>(num_heads_)) ||
        !transformer_write_u64(
            output,
            static_cast<std::uint64_t>(feed_forward_size_))) {
        return false;
    }

    return transformer_write_matrix(output, query_weight_) &&
           transformer_write_matrix(output, key_weight_) &&
           transformer_write_matrix(output, value_weight_) &&
           transformer_write_matrix(output, output_weight_) &&
           transformer_write_matrix(output, feed_forward_in_) &&
           transformer_write_matrix(output, feed_forward_out_);
}

bool TransformerBlock::load(
    std::istream& input) {

    std::uint64_t embedding_size = 0;
    std::uint64_t num_heads = 0;
    std::uint64_t feed_forward_size = 0;

    if (!transformer_read_u64(input, embedding_size) ||
        !transformer_read_u64(input, num_heads) ||
        !transformer_read_u64(input, feed_forward_size)) {
        return false;
    }

    if (embedding_size != embedding_size_ ||
        num_heads != num_heads_ ||
        feed_forward_size != feed_forward_size_) {
        return false;
    }

    return transformer_read_matrix(
               input,
               query_weight_,
               embedding_size_,
               embedding_size_) &&
           transformer_read_matrix(
               input,
               key_weight_,
               embedding_size_,
               embedding_size_) &&
           transformer_read_matrix(
               input,
               value_weight_,
               embedding_size_,
               embedding_size_) &&
           transformer_read_matrix(
               input,
               output_weight_,
               embedding_size_,
               embedding_size_) &&
           transformer_read_matrix(
               input,
               feed_forward_in_,
               embedding_size_,
               feed_forward_size_) &&
           transformer_read_matrix(
               input,
               feed_forward_out_,
               feed_forward_size_,
               embedding_size_);
}
