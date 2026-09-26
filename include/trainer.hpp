#pragma once

#include <cstddef>
#include <vector>

float ultron_cross_entropy_loss(
    const std::vector<float>& probabilities,
    std::size_t target);

std::vector<float> ultron_softmax(
    const std::vector<float>& logits);

void ultron_softmax_into(
    const std::vector<float>& logits,
    std::vector<float>& probabilities);

void ultron_output_gradient(
    const std::vector<float>& hidden,
    const std::vector<float>& probabilities,
    std::size_t target,
    std::vector<float>& gradient);
