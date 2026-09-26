#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

class TransformerBlock {
public:
    using Matrix = std::vector<std::vector<float>>;

    struct Gradients {
        Matrix query_weight;
        Matrix key_weight;
        Matrix value_weight;
        Matrix output_weight;
        Matrix feed_forward_in;
        Matrix feed_forward_out;
    };

    TransformerBlock(
        std::size_t embedding_size,
        std::size_t num_heads = 4,
        std::size_t feed_forward_size = 128);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings) const;

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<float>& attention_weights) const;

    void backward(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::vector<float>>& grad_output,
        std::vector<std::vector<float>>& grad_embeddings,
        Gradients& gradients) const;

    std::size_t parameter_count() const;

    void get_parameters(
        std::vector<float>& parameters) const;

    void set_parameters(
        const std::vector<float>& parameters);

    void flatten_gradients(
        const Gradients& gradients,
        std::vector<float>& flattened) const;

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

    static std::vector<float> layer_norm_backward(
        const std::vector<float>& normalized_input,
        const std::vector<float>& gradient);

    static float gelu(float value);
    static float gelu_derivative(float value);

    static void zero_gradients(
        Gradients& gradients,
        const Matrix& query_weight,
        const Matrix& key_weight,
        const Matrix& value_weight,
        const Matrix& output_weight,
        const Matrix& feed_forward_in,
        const Matrix& feed_forward_out);
};
