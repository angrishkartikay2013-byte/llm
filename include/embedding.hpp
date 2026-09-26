#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

class Embedding {
public:
    Embedding(
        std::size_t vocabulary_size,
        std::size_t embedding_size);

    const std::vector<float>& lookup(
        std::size_t token_id) const;

    std::vector<float>& lookup_mutable(
        std::size_t token_id);

    std::size_t vocabulary_size() const;
    std::size_t embedding_size() const;

    std::size_t parameter_count() const;

    void get_parameters(
        std::vector<float>& parameters) const;

    void set_parameters(
        const std::vector<float>& parameters);

    bool save(std::ostream& output) const;
    bool load(std::istream& input);

private:
    std::size_t vocabulary_size_;
    std::size_t embedding_size_;
    std::vector<std::vector<float>> weights_;
};
