#pragma once

#include <cstddef>
#include <string>

struct ModelConfig {
    std::size_t embedding_size = 32;
    std::size_t num_heads = 4;
    std::size_t feed_forward_size = 128;
    std::size_t max_sequence_length = 128;
    std::size_t epochs = 1;
    float learning_rate = 0.003f;
    float temperature = 0.8f;
    std::size_t top_k = 8;
    std::size_t max_new_tokens = 16;
    unsigned int seed = 42;

    bool validate(std::string& error) const;
};