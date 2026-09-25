#include "embedding.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

Embedding::Embedding(
    std::size_t vocabulary_size,
    std::size_t embedding_size)
    : vocabulary_size_(vocabulary_size),
      embedding_size_(embedding_size),
      weights_(vocabulary_size,
               std::vector<float>(embedding_size, 0.0f)) {

    std::mt19937 generator(42);
    const float limit =
        std::sqrt(6.0f / static_cast<float>(vocabulary_size + embedding_size));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    for (auto& row : weights_) {
        for (float& value : row) {
            value = distribution(generator);
        }
    }
}

const std::vector<float>& Embedding::lookup(std::size_t token_id) const {
    if (token_id >= vocabulary_size_) {
        throw std::out_of_range("Embedding token id out of range");
    }
    return weights_[token_id];
}

std::size_t Embedding::vocabulary_size() const {
    return vocabulary_size_;
}

std::size_t Embedding::embedding_size() const {
    return embedding_size_;
}
