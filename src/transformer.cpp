#include "transformer.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <istream>
#include <ostream>


namespace {

bool transformer_write_u64(
    std::ostream& output,
    std::uint64_t value) {
    output.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));
    return static_cast<bool>(output);
}

bool transformer_read_u64(
    std::istream& input,
    std::uint64_t& value) {
    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));
    return static_cast<bool>(input);
}

bool transformer_write_matrix(
    std::ostream& output,
    const TransformerBlock::Matrix& matrix) {
    if (!transformer_write_u64(
            output,
            static_cast<std::uint64_t>(matrix.size()))) {
        return false;
    }

    const std::size_t columns =
        matrix.empty() ? 0 : matrix.front().size();

    if (!transformer_write_u64(
            output,
            static_cast<std::uint64_t>(columns))) {
        return false;
    }

    for (const auto& row : matrix) {
        if (row.size() != columns) return false;

        output.write(
            reinterpret_cast<const char*>(row.data()),
            static_cast<std::streamsize>(
                row.size() * sizeof(float)));

        if (!output) return false;
    }

    return true;
}

bool transformer_read_matrix(
    std::istream& input,
    TransformerBlock::Matrix& matrix,
    std::size_t expected_rows,
    std::size_t expected_columns) {
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;

    if (!transformer_read_u64(input, rows) ||
        !transformer_read_u64(input, columns) ||
        rows != expected_rows ||
        columns != expected_columns) {
        return false;
    }

    matrix.assign(
        expected_rows,
        std::vector<float>(
            expected_columns,
            0.0f));

    for (auto& row : matrix) {
        input.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(
                row.size() * sizeof(float)));

        if (!input) return false;
    }

    return true;
}

void add_scaled(
    std::vector<float>& target,
    const std::vector<float>& source,
    float scale) {

    if (target.size() != source.size()) {
        throw std::invalid_argument(
            "Gradient shape mismatch");
    }

    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] += source[i] * scale;
    }
}

void matrix_gradient(
    TransformerBlock::Matrix& gradient,
    const std::vector<float>& input,
    const std::vector<float>& output_gradient) {

    if (gradient.size() != input.size()) {
        throw std::invalid_argument(
            "Matrix gradient row mismatch");
    }

    for (std::size_t row = 0;
         row < input.size();
         ++row) {
        for (std::size_t column = 0;
             column < output_gradient.size();
             ++column) {
            gradient[row][column] +=
                input[row] * output_gradient[column];
        }
    }
}

std::vector<float> matrix_input_gradient(
    const TransformerBlock::Matrix& weights,
    const std::vector<float>& output_gradient) {

    if (weights.empty() ||
        weights.front().size() != output_gradient.size()) {
        throw std::invalid_argument(
            "Matrix input gradient shape mismatch");
    }

    std::vector<float> result(weights.size(), 0.0f);

    for (std::size_t row = 0;
         row < weights.size();
         ++row) {
        for (std::size_t column = 0;
             column < output_gradient.size();
             ++column) {
            result[row] +=
                output_gradient[column] *
                weights[row][column];
        }
    }

    return result;
}

std::vector<float> matrix_flatten(
    const TransformerBlock::Matrix& matrix) {

    std::size_t count = 0;
    for (const auto& row : matrix) {
        count += row.size();
    }

    std::vector<float> result;
    result.reserve(count);

    for (const auto& row : matrix) {
        result.insert(
            result.end(),
            row.begin(),
            row.end());
    }

    return result;
}

void matrix_unflatten(
    TransformerBlock::Matrix& matrix,
    const std::vector<float>& parameters,
    std::size_t& offset) {

    for (auto& row : matrix) {
        for (float& value : row) {
            if (offset >= parameters.size()) {
                throw std::invalid_argument(
                    "Transformer parameter buffer is too small");
            }
            value = parameters[offset++];
        }
    }
}

} // namespace

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
    std::size_t feed_forward_size,
    unsigned int seed)
    : embedding_size_(embedding_size),
      num_heads_(num_heads),
      head_size_(embedding_size / num_heads),
      feed_forward_size_(feed_forward_size),
      query_weight_(random_matrix(embedding_size, embedding_size, seed)),
      key_weight_(random_matrix(embedding_size, embedding_size, seed + 1)),
      value_weight_(random_matrix(embedding_size, embedding_size, seed + 2)),
      output_weight_(random_matrix(embedding_size, embedding_size, seed + 3)),
      feed_forward_in_(random_matrix(embedding_size, feed_forward_size, seed + 4)),
      feed_forward_out_(random_matrix(feed_forward_size, embedding_size, seed + 5)) {

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

    if (weights.empty()) {
        return {};
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

std::vector<float> TransformerBlock::layer_norm_backward(
    const std::vector<float>& normalized_input,
    const std::vector<float>& gradient) {

    if (normalized_input.size() != gradient.size()) {
        throw std::invalid_argument(
            "LayerNorm gradient shape mismatch");
    }

    if (normalized_input.empty()) {
        return {};
    }

    float mean = 0.0f;
    for (float value : normalized_input) {
        mean += value;
    }
    mean /= static_cast<float>(normalized_input.size());

    float variance = 0.0f;
    for (float value : normalized_input) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<float>(normalized_input.size());

    constexpr float epsilon = 1e-5f;
    const float inverse_std =
        1.0f / std::sqrt(variance + epsilon);

    std::vector<float> x_hat(normalized_input.size());
    for (std::size_t i = 0; i < normalized_input.size(); ++i) {
        x_hat[i] =
            (normalized_input[i] - mean) *
            inverse_std;
    }

    float sum_gradient = 0.0f;
    float sum_gradient_xhat = 0.0f;

    for (std::size_t i = 0; i < gradient.size(); ++i) {
        sum_gradient += gradient[i];
        sum_gradient_xhat += gradient[i] * x_hat[i];
    }

    const float count =
        static_cast<float>(gradient.size());

    std::vector<float> result(gradient.size(), 0.0f);

    for (std::size_t i = 0; i < gradient.size(); ++i) {
        result[i] =
            (inverse_std / count) *
            (count * gradient[i] -
             sum_gradient -
             x_hat[i] * sum_gradient_xhat);
    }

    return result;
}

float TransformerBlock::gelu(float value) {
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    return 0.5f * value *
        (1.0f + std::tanh(
            kSqrtTwoOverPi *
            (value + 0.044715f * value * value * value)));
}

float TransformerBlock::gelu_derivative(float value) {
    constexpr float kSqrtTwoOverPi = 0.7978845608f;
    constexpr float kGeluCoefficient = 0.044715f;

    const float cubic =
        value * value * value;

    const float inner =
        kSqrtTwoOverPi *
        (value + kGeluCoefficient * cubic);

    const float tanh_inner =
        std::tanh(inner);

    const float inner_derivative =
        kSqrtTwoOverPi *
        (1.0f +
         3.0f * kGeluCoefficient *
         value * value);

    return 0.5f *
        (1.0f + tanh_inner) +
        0.5f *
        value *
        (1.0f - tanh_inner * tanh_inner) *
        inner_derivative;
}

void TransformerBlock::zero_gradients(
    Gradients& gradients,
    const Matrix& query_weight,
    const Matrix& key_weight,
    const Matrix& value_weight,
    const Matrix& output_weight,
    const Matrix& feed_forward_in,
    const Matrix& feed_forward_out) {

    gradients.query_weight =
        Matrix(
            query_weight.size(),
            std::vector<float>(
                query_weight.empty()
                    ? 0
                    : query_weight.front().size(),
                0.0f));

    gradients.key_weight =
        Matrix(
            key_weight.size(),
            std::vector<float>(
                key_weight.empty()
                    ? 0
                    : key_weight.front().size(),
                0.0f));

    gradients.value_weight =
        Matrix(
            value_weight.size(),
            std::vector<float>(
                value_weight.empty()
                    ? 0
                    : value_weight.front().size(),
                0.0f));

    gradients.output_weight =
        Matrix(
            output_weight.size(),
            std::vector<float>(
                output_weight.empty()
                    ? 0
                    : output_weight.front().size(),
                0.0f));

    gradients.feed_forward_in =
        Matrix(
            feed_forward_in.size(),
            std::vector<float>(
                feed_forward_in.empty()
                    ? 0
                    : feed_forward_in.front().size(),
                0.0f));

    gradients.feed_forward_out =
        Matrix(
            feed_forward_out.size(),
            std::vector<float>(
                feed_forward_out.empty()
                    ? 0
                    : feed_forward_out.front().size(),
                0.0f));
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

    const std::size_t sequence_length =
        embeddings.size();

    std::vector<std::vector<float>> queries(
        sequence_length);
    std::vector<std::vector<float>> keys(
        sequence_length);
    std::vector<std::vector<float>> values(
        sequence_length);

    const std::size_t qkv_threads =
        transformer_thread_count(sequence_length);

    std::vector<std::thread> qkv_workers;
    qkv_workers.reserve(qkv_threads);

    for (std::size_t worker = 0;
         worker < qkv_threads;
         ++worker) {

        qkv_workers.emplace_back(
            [&, worker]() {
                const std::size_t begin =
                    (sequence_length * worker) / qkv_threads;
                const std::size_t end =
                    (sequence_length * (worker + 1)) / qkv_threads;

                for (std::size_t i = begin;
                     i < end;
                     ++i) {
                    queries[i] =
                        linear(embeddings[i], query_weight_);
                    keys[i] =
                        linear(embeddings[i], key_weight_);
                    values[i] =
                        linear(embeddings[i], value_weight_);
                }
            });
    }

    for (auto& worker : qkv_workers) {
        worker.join();
    }

    std::vector<std::vector<std::vector<float>>> attention_weights(
        sequence_length,
        std::vector<std::vector<float>>(
            num_heads_));

    std::vector<std::vector<float>> attention_concat(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(head_size_));

    const std::size_t attention_threads =
        transformer_thread_count(sequence_length);

    std::vector<std::thread> attention_workers;
    attention_workers.reserve(attention_threads);

    for (std::size_t worker = 0;
         worker < attention_threads;
         ++worker) {

        attention_workers.emplace_back(
            [&, worker]() {
                const std::size_t begin =
                    (sequence_length * worker) / attention_threads;
                const std::size_t end =
                    (sequence_length * (worker + 1)) / attention_threads;

                for (std::size_t query = begin;
                     query < end;
                     ++query) {

                    for (std::size_t head = 0;
                         head < num_heads_;
                         ++head) {

                        const std::size_t offset =
                            head * head_size_;

                        std::vector<float> scores(
                            query + 1,
                            0.0f);

                        float max_score =
                            -std::numeric_limits<float>::infinity();

                        for (std::size_t key = 0;
                             key <= query;
                             ++key) {

                            float dot = 0.0f;

                            for (std::size_t d = 0;
                                 d < head_size_;
                                 ++d) {
                                dot +=
                                    queries[query][offset + d] *
                                    keys[key][offset + d];
                            }

                            scores[key] =
                                dot * scale;

                            max_score =
                                std::max(
                                    max_score,
                                    scores[key]);
                        }

                        float sum = 0.0f;

                        for (float& score : scores) {
                            score =
                                std::exp(
                                    score - max_score);
                            sum += score;
                        }

                        if (sum <= 0.0f) {
                            throw std::runtime_error(
                                "Attention softmax normalization failed");
                        }

                        attention_weights[query][head].resize(
                            query + 1);

                        for (std::size_t key = 0;
                             key <= query;
                             ++key) {

                            const float weight =
                                scores[key] / sum;

                            attention_weights[query][head][key] =
                                weight;

                            for (std::size_t d = 0;
                                 d < head_size_;
                                 ++d) {
                                attention_concat[query][offset + d] +=
                                    weight *
                                    values[key][offset + d];
                            }
                        }
                    }
                }
            });
    }

    for (auto& worker : attention_workers) {
        worker.join();
    }

    std::vector<std::vector<float>> attention_residual(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    std::vector<std::vector<float>> norm1(
        sequence_length);

    for (std::size_t query = 0;
         query < sequence_length;
         ++query) {

        const auto projected =
            linear(
                attention_concat[query],
                output_weight_);

        attention_residual[query] =
            embeddings[query];

        for (std::size_t dimension = 0;
             dimension < embedding_size_;
             ++dimension) {
            attention_residual[query][dimension] +=
                projected[dimension];
        }

        norm1[query] =
            layer_norm(
                attention_residual[query]);
    }

    std::vector<std::vector<float>> ffn_pre(
        sequence_length);
    std::vector<std::vector<float>> ffn_activated(
        sequence_length);
    std::vector<std::vector<float>> ffn_projected(
        sequence_length);
    std::vector<std::vector<float>> ffn_residual(
        sequence_length);
    std::vector<std::vector<float>> output(
        sequence_length);

    const std::size_t ffn_threads =
        transformer_thread_count(sequence_length);

    std::vector<std::thread> ffn_workers;
    ffn_workers.reserve(ffn_threads);

    for (std::size_t worker = 0;
         worker < ffn_threads;
         ++worker) {

        ffn_workers.emplace_back(
            [&, worker]() {
                const std::size_t begin =
                    (sequence_length * worker) / ffn_threads;
                const std::size_t end =
                    (sequence_length * (worker + 1)) / ffn_threads;

                for (std::size_t i = begin;
                     i < end;
                     ++i) {

                    ffn_pre[i] =
                        linear(
                            norm1[i],
                            feed_forward_in_);

                    ffn_activated[i].resize(
                        feed_forward_size_);

                    for (std::size_t j = 0;
                         j < feed_forward_size_;
                         ++j) {
                        ffn_activated[i][j] =
                            gelu(
                                ffn_pre[i][j]);
                    }

                    ffn_projected[i] =
                        linear(
                            ffn_activated[i],
                            feed_forward_out_);

                    ffn_residual[i] =
                        norm1[i];

                    for (std::size_t dimension = 0;
                         dimension < embedding_size_;
                         ++dimension) {
                        ffn_residual[i][dimension] +=
                            ffn_projected[i][dimension];
                    }

                    output[i] =
                        layer_norm(
                            ffn_residual[i]);
                }
            });
    }

    for (auto& worker : ffn_workers) {
        worker.join();
    }

    return output;
}

void TransformerBlock::backward(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<std::vector<float>>& grad_output,
    std::vector<std::vector<float>>& grad_embeddings,
    Gradients& gradients) const {

    if (embeddings.empty()) {
        grad_embeddings.clear();
        zero_gradients(
            gradients,
            query_weight_,
            key_weight_,
            value_weight_,
            output_weight_,
            feed_forward_in_,
            feed_forward_out_);
        return;
    }

    if (grad_output.size() != embeddings.size()) {
        throw std::invalid_argument(
            "Transformer backward sequence mismatch");
    }

    for (const auto& row : embeddings) {
        if (row.size() != embedding_size_) {
            throw std::invalid_argument(
                "Transformer backward embedding mismatch");
        }
    }

    for (const auto& row : grad_output) {
        if (row.size() != embedding_size_) {
            throw std::invalid_argument(
                "Transformer backward gradient mismatch");
        }
    }

    zero_gradients(
        gradients,
        query_weight_,
        key_weight_,
        value_weight_,
        output_weight_,
        feed_forward_in_,
        feed_forward_out_);

    const std::size_t sequence_length =
        embeddings.size();

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(head_size_));

    std::vector<std::vector<float>> queries(
        sequence_length);
    std::vector<std::vector<float>> keys(
        sequence_length);
    std::vector<std::vector<float>> values(
        sequence_length);

    std::vector<std::vector<std::vector<float>>> attention_weights(
        sequence_length,
        std::vector<std::vector<float>>(
            num_heads_));

    std::vector<std::vector<float>> attention_concat(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    const std::size_t backward_qkv_threads =
        transformer_thread_count(sequence_length);

    std::vector<std::thread> backward_qkv_workers;
    backward_qkv_workers.reserve(backward_qkv_threads);

    for (std::size_t worker = 0;
         worker < backward_qkv_threads;
         ++worker) {

        backward_qkv_workers.emplace_back(
            [&, worker]() {
                const std::size_t begin =
                    (sequence_length * worker) / backward_qkv_threads;
                const std::size_t end =
                    (sequence_length * (worker + 1)) / backward_qkv_threads;

                for (std::size_t i = begin;
                     i < end;
                     ++i) {
                    queries[i] =
                        linear(
                            embeddings[i],
                            query_weight_);
                    keys[i] =
                        linear(
                            embeddings[i],
                            key_weight_);
                    values[i] =
                        linear(
                            embeddings[i],
                            value_weight_);
                }
            });
    }

    for (auto& worker : backward_qkv_workers) {
        worker.join();
    }

    for (std::size_t query = 0;
         query < sequence_length;
         ++query) {

        for (std::size_t head = 0;
             head < num_heads_;
             ++head) {

            const std::size_t offset =
                head * head_size_;

            std::vector<float> scores(
                query + 1,
                0.0f);

            float max_score =
                -std::numeric_limits<float>::infinity();

            for (std::size_t key = 0;
                 key <= query;
                 ++key) {

                float dot = 0.0f;

                for (std::size_t d = 0;
                     d < head_size_;
                     ++d) {
                    dot +=
                        queries[query][offset + d] *
                        keys[key][offset + d];
                }

                scores[key] =
                    dot * scale;

                max_score =
                    std::max(
                        max_score,
                        scores[key]);
            }

            float sum = 0.0f;

            for (float& score : scores) {
                score =
                    std::exp(
                        score - max_score);
                sum += score;
            }

            attention_weights[query][head].resize(
                query + 1);

            for (std::size_t key = 0;
                 key <= query;
                 ++key) {
                attention_weights[query][head][key] =
                    scores[key] / sum;

                for (std::size_t d = 0;
                     d < head_size_;
                     ++d) {
                    attention_concat[query][offset + d] +=
                        attention_weights[query][head][key] *
                        values[key][offset + d];
                }
            }
        }
    }

    std::vector<std::vector<float>> attention_residual(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    std::vector<std::vector<float>> norm1(
        sequence_length);

    for (std::size_t i = 0;
         i < sequence_length;
         ++i) {

        const auto projected =
            linear(
                attention_concat[i],
                output_weight_);

        attention_residual[i] =
            embeddings[i];

        for (std::size_t dimension = 0;
             dimension < embedding_size_;
             ++dimension) {
            attention_residual[i][dimension] +=
                projected[dimension];
        }

        norm1[i] =
            layer_norm(
                attention_residual[i]);
    }

    std::vector<std::vector<float>> ffn_pre(
        sequence_length);
    std::vector<std::vector<float>> ffn_activated(
        sequence_length);
    std::vector<std::vector<float>> ffn_projected(
        sequence_length);
    std::vector<std::vector<float>> ffn_residual(
        sequence_length);
    std::vector<std::vector<float>> grad_norm1(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));
    std::vector<std::vector<float>> grad_attention_projected(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));
    std::vector<std::vector<float>> grad_attention_concat(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    for (std::size_t i = 0;
         i < sequence_length;
         ++i) {

        ffn_pre[i] =
            linear(
                norm1[i],
                feed_forward_in_);

        ffn_activated[i].resize(
            feed_forward_size_);

        for (std::size_t j = 0;
             j < feed_forward_size_;
             ++j) {
            ffn_activated[i][j] =
                gelu(
                    ffn_pre[i][j]);
        }

        ffn_projected[i] =
            linear(
                ffn_activated[i],
                feed_forward_out_);

        ffn_residual[i] =
            norm1[i];

        for (std::size_t dimension = 0;
             dimension < embedding_size_;
             ++dimension) {
            ffn_residual[i][dimension] +=
                ffn_projected[i][dimension];
        }

        const auto grad_residual2 =
            layer_norm_backward(
                ffn_residual[i],
                grad_output[i]);

        add_scaled(
            grad_norm1[i],
            grad_residual2,
            1.0f);

        matrix_gradient(
            gradients.feed_forward_out,
            ffn_activated[i],
            grad_residual2);

        const auto grad_activated =
            matrix_input_gradient(
                feed_forward_out_,
                grad_residual2);

        std::vector<float> grad_pre(
            feed_forward_size_,
            0.0f);

        for (std::size_t j = 0;
             j < feed_forward_size_;
             ++j) {
            grad_pre[j] =
                grad_activated[j] *
                gelu_derivative(
                    ffn_pre[i][j]);
        }

        matrix_gradient(
            gradients.feed_forward_in,
            norm1[i],
            grad_pre);

        const auto grad_norm1_from_ffn =
            matrix_input_gradient(
                feed_forward_in_,
                grad_pre);

        add_scaled(
            grad_norm1[i],
            grad_norm1_from_ffn,
            1.0f);
    }

    std::vector<std::vector<float>> grad_embeddings_internal(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    std::vector<std::vector<float>> grad_queries(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    std::vector<std::vector<float>> grad_keys(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    std::vector<std::vector<float>> grad_values(
        sequence_length,
        std::vector<float>(
            embedding_size_,
            0.0f));

    for (std::size_t i = 0;
         i < sequence_length;
         ++i) {

        const auto grad_residual1 =
            layer_norm_backward(
                attention_residual[i],
                grad_norm1[i]);

        add_scaled(
            grad_embeddings_internal[i],
            grad_residual1,
            1.0f);

        grad_attention_projected[i] =
            grad_residual1;

        matrix_gradient(
            gradients.output_weight,
            attention_concat[i],
            grad_attention_projected[i]);

        grad_attention_concat[i] =
            matrix_input_gradient(
                output_weight_,
                grad_attention_projected[i]);
    }

    for (std::size_t query = 0;
         query < sequence_length;
         ++query) {

        for (std::size_t head = 0;
             head < num_heads_;
             ++head) {

            const std::size_t offset =
                head * head_size_;

            const auto& weights =
                attention_weights[query][head];

            std::vector<float> grad_weights(
                weights.size(),
                0.0f);

            for (std::size_t key = 0;
                 key <= query;
                 ++key) {

                float gradient_weight = 0.0f;

                for (std::size_t d = 0;
                     d < head_size_;
                     ++d) {
                    gradient_weight +=
                        grad_attention_concat[query][offset + d] *
                        values[key][offset + d];

                    grad_values[key][offset + d] +=
                        weights[key] *
                        grad_attention_concat[query][offset + d];
                }

                grad_weights[key] =
                    gradient_weight;
            }

            float weighted_gradient_sum = 0.0f;

            for (std::size_t key = 0;
                 key <= query;
                 ++key) {
                weighted_gradient_sum +=
                    grad_weights[key] *
                    weights[key];
            }

            for (std::size_t key = 0;
                 key <= query;
                 ++key) {

                const float grad_score =
                    weights[key] *
                    (grad_weights[key] -
                     weighted_gradient_sum);

                for (std::size_t d = 0;
                     d < head_size_;
                     ++d) {

                    grad_queries[query][offset + d] +=
                        grad_score *
                        keys[key][offset + d] *
                        scale;

                    grad_keys[key][offset + d] +=
                        grad_score *
                        queries[query][offset + d] *
                        scale;
                }
            }
        }
    }

    for (std::size_t i = 0;
         i < sequence_length;
         ++i) {

        matrix_gradient(
            gradients.query_weight,
            embeddings[i],
            grad_queries[i]);

        matrix_gradient(
            gradients.key_weight,
            embeddings[i],
            grad_keys[i]);

        matrix_gradient(
            gradients.value_weight,
            embeddings[i],
            grad_values[i]);

        const auto grad_from_query =
            matrix_input_gradient(
                query_weight_,
                grad_queries[i]);

        const auto grad_from_key =
            matrix_input_gradient(
                key_weight_,
                grad_keys[i]);

        const auto grad_from_value =
            matrix_input_gradient(
                value_weight_,
                grad_values[i]);

        add_scaled(
            grad_embeddings_internal[i],
            grad_from_query,
            1.0f);

        add_scaled(
            grad_embeddings_internal[i],
            grad_from_key,
            1.0f);

        add_scaled(
            grad_embeddings_internal[i],
            grad_from_value,
            1.0f);
    }

    grad_embeddings =
        std::move(
            grad_embeddings_internal);
}

std::vector<std::vector<float>> TransformerBlock::forward(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<float>& attention_weights) const {

    (void)attention_weights;
    return forward(embeddings);
}

std::size_t TransformerBlock::parameter_count() const {
    return query_weight_.size() * embedding_size_ +
           key_weight_.size() * embedding_size_ +
           value_weight_.size() * embedding_size_ +
           output_weight_.size() * embedding_size_ +
           feed_forward_in_.size() * feed_forward_size_ +
           feed_forward_out_.size() * embedding_size_;
}

void TransformerBlock::get_parameters(
    std::vector<float>& parameters) const {

    parameters.clear();

    const auto append =
        [&](const Matrix& matrix) {
            const auto flat =
                matrix_flatten(matrix);
            parameters.insert(
                parameters.end(),
                flat.begin(),
                flat.end());
        };

    parameters.reserve(parameter_count());

    append(query_weight_);
    append(key_weight_);
    append(value_weight_);
    append(output_weight_);
    append(feed_forward_in_);
    append(feed_forward_out_);
}

void TransformerBlock::set_parameters(
    const std::vector<float>& parameters) {

    if (parameters.size() != parameter_count()) {
        throw std::invalid_argument(
            "Transformer parameter size mismatch");
    }

    std::size_t offset = 0;

    matrix_unflatten(
        query_weight_,
        parameters,
        offset);

    matrix_unflatten(
        key_weight_,
        parameters,
        offset);

    matrix_unflatten(
        value_weight_,
        parameters,
        offset);

    matrix_unflatten(
        output_weight_,
        parameters,
        offset);

    matrix_unflatten(
        feed_forward_in_,
        parameters,
        offset);

    matrix_unflatten(
        feed_forward_out_,
        parameters,
        offset);
}

void TransformerBlock::flatten_gradients(
    const Gradients& gradients,
    std::vector<float>& flattened) const {

    flattened.clear();
    flattened.reserve(parameter_count());

    const auto append =
        [&](const Matrix& matrix) {
            const auto flat =
                matrix_flatten(matrix);
            flattened.insert(
                flattened.end(),
                flat.begin(),
                flat.end());
        };

    append(gradients.query_weight);
    append(gradients.key_weight);
    append(gradients.value_weight);
    append(gradients.output_weight);
    append(gradients.feed_forward_in);
    append(gradients.feed_forward_out);
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
