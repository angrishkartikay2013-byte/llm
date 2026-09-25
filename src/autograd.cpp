#include <cstddef>
#include <vector>

void ultron_cross_entropy_gradient(
    const std::vector<float>& probabilities,
    std::size_t target,
    std::vector<float>& gradient) {

    gradient = probabilities;

    if (target < gradient.size()) {
        gradient[target] -= 1.0f;
    }
}
