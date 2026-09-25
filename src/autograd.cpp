#include <vector>

// Lightweight gradient utility. The full computation graph is introduced
// once trainable tensor operations are connected to the model.
void ultron_cross_entropy_gradient(
    const std::vector<float>& probabilities,
    std::size_t target,
    std::vector<float>& gradient) {

    gradient = probabilities;

    if (target < gradient.size()) {
        gradient[target] -= 1.0f;
    }
}
