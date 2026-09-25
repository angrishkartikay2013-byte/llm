#include "tokenizer.hpp"

#include <cmath>
#include <vector>

// Initial next-token loss helper.
// A full optimizer-backed training loop will consume these primitives.
float ultron_cross_entropy_loss(
    const std::vector<float>& probabilities,
    std::size_t target) {

    if (target >= probabilities.size()) {
        return 0.0f;
    }

    constexpr float epsilon = 1e-8f;
    return -std::log(std::max(probabilities[target], epsilon));
}

void ultron_prepare_training_text(
    Tokenizer& tokenizer,
    const std::string& text) {

    tokenizer.train(text);
}
