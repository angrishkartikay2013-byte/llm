#pragma once

#include <cstddef>
#include <vector>

class Embedding {
public:
    Embedding(std::size_t vocabulary_size, std::size_t embedding_size);

    const std::vector<float>& lookup(std::size_t token_id) const;

    std::size_t vocabulary_size() const;
    std::size_t embedding_size() const;

private:
    std::size_t vocabulary_size_;
    std::size_t embedding_size_;
    std::vector<std::vector<float>> weights_;
};