#include "config.hpp"

#include <sstream>

bool ModelConfig::validate(std::string& error) const {
    if (embedding_size == 0) {
        error = "embedding_size must be greater than zero";
        return false;
    }
    if (num_heads == 0 || embedding_size % num_heads != 0) {
        error = "embedding_size must be divisible by num_heads";
        return false;
    }
    if (feed_forward_size == 0) {
        error = "feed_forward_size must be greater than zero";
        return false;
    }
    if (max_sequence_length == 0) {
        error = "max_sequence_length must be greater than zero";
        return false;
    }
    if (learning_rate <= 0.0f) {
        error = "learning_rate must be positive";
        return false;
    }
    if (temperature <= 0.0f) {
        error = "temperature must be positive";
        return false;
    }
    return true;
}
