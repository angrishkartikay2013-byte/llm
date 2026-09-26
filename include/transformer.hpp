#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

class TransformerBlock {
public:
    using Matrix = std::vector<std::vector<float>>;

    TransformerBlock(
        std::size_t embedding_size,
        std::size_t num_heads = 4,
        std::size_t feed_forward_size = 128);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings) const;

    // Compatibility wrapper for the initial engine API.
    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<float>& attention_weights) const;

    bool save(std::ostream& output) const;
    bool load(std::istream& input);

private:
    std::size_t embedding_size_;
    std::size_t num_heads_;
    std::size_t head_size_;
    std::size_t feed_forward_size_;

    Matrix query_weight_;
    Matrix key_weight_;
    Matrix value_weight_;
    Matrix output_weight_;

    Matrix feed_forward_in_;
    Matrix feed_forward_out_;

    static Matrix random_matrix(
        std::size_t rows,
        std::size_t cols,
        unsigned int seed);

    static std::vector<float> linear(
        const std::vector<float>& input,
        const Matrix& weights);

    static std::vector<float> layer_norm(
        const std::vector<float>& input);

    static float gelu(float value);
};