#pragma once

#include <vector>

class Attention {
public:
    Attention();

    std::vector<float> softmax(
        const std::vector<float>& scores) const;

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& embeddings) const;
};