#pragma once

#include <vector>

class TransformerBlock {
public:
    explicit TransformerBlock(int embedding_size);

    std::vector<float> feed_forward(
        const std::vector<float>& input) const;

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<float>& attention_weights) const;

private:
    int embedding_size_;
};