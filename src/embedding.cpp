#include "embedding.hpp"

#include <cstdint>
#include <istream>
#include <ostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace {
bool write_u64(std::ostream& output, std::uint64_t value) { output.write(reinterpret_cast<const char*>(&value), sizeof(value)); return static_cast<bool>(output); }
bool read_u64(std::istream& input, std::uint64_t& value) { input.read(reinterpret_cast<char*>(&value), sizeof(value)); return static_cast<bool>(input); }
}

Embedding::Embedding(std::size_t vocabulary_size, std::size_t embedding_size)
    : vocabulary_size_(vocabulary_size), embedding_size_(embedding_size), weights_(vocabulary_size, std::vector<float>(embedding_size, 0.0f)) {
    if (vocabulary_size == 0 || embedding_size == 0) throw std::invalid_argument("Embedding dimensions must be non-zero");
    std::mt19937 generator(42);
    const float limit = std::sqrt(6.0f / static_cast<float>(vocabulary_size + embedding_size));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    for (auto& row : weights_) for (float& value : row) value = distribution(generator);
}

const std::vector<float>& Embedding::lookup(std::size_t token_id) const {
    if (token_id >= vocabulary_size_) throw std::out_of_range("Embedding token id out of range");
    return weights_[token_id];
}
std::size_t Embedding::vocabulary_size() const { return vocabulary_size_; }
std::size_t Embedding::embedding_size() const { return embedding_size_; }

bool Embedding::save(std::ostream& output) const {
    if (!write_u64(output, static_cast<std::uint64_t>(vocabulary_size_)) || !write_u64(output, static_cast<std::uint64_t>(embedding_size_))) return false;
    for (const auto& row : weights_) {
        output.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!output) return false;
    }
    return true;
}

bool Embedding::load(std::istream& input) {
    std::uint64_t vocabulary = 0, dimension = 0;
    if (!read_u64(input, vocabulary) || !read_u64(input, dimension)) return false;
    if (vocabulary == 0 || vocabulary > 10000000 || dimension == 0 || dimension > 16384) return false;
    std::vector<std::vector<float>> loaded(static_cast<std::size_t>(vocabulary), std::vector<float>(static_cast<std::size_t>(dimension), 0.0f));
    for (auto& row : loaded) {
        input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!input) return false;
    }
    vocabulary_size_ = static_cast<std::size_t>(vocabulary);
    embedding_size_ = static_cast<std::size_t>(dimension);
    weights_ = std::move(loaded);
    return true;
}
