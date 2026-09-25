#include <algorithm>
#include <cmath>
#include <vector>

// Standalone LayerNorm implementation used by later transformer blocks.
std::vector<float> ultron_layer_norm(
    const std::vector<float>& input,
    float epsilon) {

    if (input.empty()) {
        return {};
    }

    float mean = 0.0f;
    for (float value : input) {
        mean += value;
    }
    mean /= static_cast<float>(input.size());

    float variance = 0.0f;
    for (float value : input) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<float>(input.size());

    const float inverse_std =
        1.0f / std::sqrt(variance + epsilon);

    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (input[i] - mean) * inverse_std;
    }

    return output;
}
