#include "attention.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace {
std::vector<float> scaled_head_scores(
    const std::vector<std::vector<float>>& x,
    std::size_t query,
    std::size_t start,
    std::size_t head_size) {

    std::vector<float> scores(x.size(), -INFINITY);
    for (std::size_t key = 0; key <= query; ++key) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < head_size; ++d) {
            dot += x[query][start + d] * x[key][start + d];
        }
        scores[key] =
            dot / std::sqrt(static_cast<float>(head_size));
    }
    return scores;
}
}

// Initial multi-head attention utility.
// The public interface remains intentionally small while the model
// architecture is expanded in later training/runtime commits.
void ultron_multi_head_attention_placeholder(
    const std::vector<std::vector<float>>& embeddings,
    std::size_t heads) {

    if (embeddings.empty() || heads == 0) {
        return;
    }

    const std::size_t dimension = embeddings.front().size();
    if (dimension == 0 || dimension % heads != 0) {
        throw std::invalid_argument(
            "Embedding size must be divisible by the head count");
    }

    const std::size_t head_size = dimension / heads;
    for (std::size_t query = 0; query < embeddings.size(); ++query) {
        for (std::size_t head = 0; head < heads; ++head) {
            (void)scaled_head_scores(
                embeddings, query, head * head_size, head_size);
        }
    }
}
