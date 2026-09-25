#include "optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

AdamOptimizer::AdamOptimizer(
    std::size_t parameter_count,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      step_count_(0),
      first_moment_(parameter_count, 0.0f),
      second_moment_(parameter_count, 0.0f) {}

void AdamOptimizer::step(
    std::vector<float>& weights,
    const std::vector<float>& gradients) {

    if (weights.size() != gradients.size() ||
        weights.size() != first_moment_.size()) {
        throw std::invalid_argument(
            "Adam parameter and gradient sizes do not match");
    }

    ++step_count_;

    const float bias1 =
        1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    const float bias2 =
        1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (std::size_t i = 0; i < weights.size(); ++i) {
        const float gradient = gradients[i];

        first_moment_[i] =
            beta1_ * first_moment_[i] +
            (1.0f - beta1_) * gradient;

        second_moment_[i] =
            beta2_ * second_moment_[i] +
            (1.0f - beta2_) * gradient * gradient;

        const float m_hat = first_moment_[i] / bias1;
        const float v_hat = second_moment_[i] / bias2;

        weights[i] -= learning_rate_ *
            m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}
