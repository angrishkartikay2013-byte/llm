#pragma once

#include <cstddef>
#include <string>

struct ModelConfig {
    std::size_t embedding_size = 128;
    std::size_t num_heads = 8;
    std::size_t feed_forward_size = 512;
    std::size_t transformer_layers = 4;
    std::size_t max_sequence_length = 256;
    std::size_t epochs = 1;
    float learning_rate = 0.001f;
    float temperature = 0.7f;
    std::size_t top_k = 20;
    std::size_t max_new_tokens = 16;
    unsigned int seed = 42;

    bool validate(std::string& error) const;
};