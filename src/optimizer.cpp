#include "optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace {
bool write_u64(
    std::ostream& output,
    std::uint64_t value) {

    output.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));

    return static_cast<bool>(output);
}

bool read_u64(
    std::istream& input,
    std::uint64_t& value) {

    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));

    return static_cast<bool>(input);
}

bool write_float(
    std::ostream& output,
    float value) {

    output.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));

    return static_cast<bool>(output);
}

bool read_float(
    std::istream& input,
    float& value) {

    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));

    return static_cast<bool>(input);
}

bool save_vector(
    std::ostream& output,
    const std::vector<float>& values) {

    if (!write_u64(
            output,
            static_cast<std::uint64_t>(
                values.size()))) {
        return false;
    }

    if (values.empty()) {
        return true;
    }

    output.write(
        reinterpret_cast<const char*>(
            values.data()),
        static_cast<std::streamsize>(
            values.size() * sizeof(float)));

    return static_cast<bool>(output);
}

bool load_vector(
    std::istream& input,
    std::vector<float>& values,
    std::size_t expected_size) {

    std::uint64_t size = 0;

    if (!read_u64(input, size) ||
        size != expected_size) {
        return false;
    }

    values.assign(
        expected_size,
        0.0f);

    if (expected_size == 0) {
        return true;
    }

    input.read(
        reinterpret_cast<char*>(
            values.data()),
        static_cast<std::streamsize>(
            expected_size * sizeof(float)));

    return static_cast<bool>(input);
}
}

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
      second_moment_(parameter_count, 0.0f) {

    if (learning_rate_ <= 0.0f ||
        beta1_ < 0.0f || beta1_ >= 1.0f ||
        beta2_ < 0.0f || beta2_ >= 1.0f ||
        epsilon_ <= 0.0f) {
        throw std::invalid_argument(
            "Invalid Adam optimizer configuration");
    }
}

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
        1.0f - std::pow(
            beta1_,
            static_cast<float>(step_count_));

    const float bias2 =
        1.0f - std::pow(
            beta2_,
            static_cast<float>(step_count_));

    if (bias1 <= 0.0f || bias2 <= 0.0f) {
        throw std::runtime_error(
            "Adam bias correction became invalid");
    }

    for (std::size_t i = 0;
         i < weights.size();
         ++i) {

        const float gradient =
            std::isfinite(gradients[i])
                ? gradients[i]
                : 0.0f;

        first_moment_[i] =
            beta1_ * first_moment_[i] +
            (1.0f - beta1_) * gradient;

        second_moment_[i] =
            beta2_ * second_moment_[i] +
            (1.0f - beta2_) * gradient * gradient;

        const float m_hat =
            first_moment_[i] / bias1;

        const float v_hat =
            second_moment_[i] / bias2;

        weights[i] -= learning_rate_ *
            m_hat /
            (std::sqrt(std::max(v_hat, 0.0f)) +
             epsilon_);

        if (!std::isfinite(weights[i])) {
            throw std::runtime_error(
                "Adam produced a non-finite parameter");
        }
    }
}

void AdamOptimizer::set_learning_rate(
    float learning_rate) {

    if (learning_rate <= 0.0f ||
        !std::isfinite(learning_rate)) {
        throw std::invalid_argument(
            "Adam learning rate must be finite and positive");
    }

    learning_rate_ = learning_rate;
}

float AdamOptimizer::learning_rate() const {
    return learning_rate_;
}

std::size_t AdamOptimizer::parameter_count() const {
    return first_moment_.size();
}

std::size_t AdamOptimizer::step_count() const {
    return step_count_;
}

bool AdamOptimizer::save(
    std::ostream& output) const {

    if (!write_float(output, learning_rate_) ||
        !write_float(output, beta1_) ||
        !write_float(output, beta2_) ||
        !write_float(output, epsilon_) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                step_count_)) ||
        !save_vector(output, first_moment_) ||
        !save_vector(output, second_moment_)) {
        return false;
    }

    return true;
}

bool AdamOptimizer::load(
    std::istream& input) {

    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    std::uint64_t step_count = 0;

    if (!read_float(input, learning_rate) ||
        !read_float(input, beta1) ||
        !read_float(input, beta2) ||
        !read_float(input, epsilon) ||
        !read_u64(input, step_count)) {
        return false;
    }

    if (!std::isfinite(learning_rate) ||
        learning_rate <= 0.0f ||
        !std::isfinite(beta1) ||
        beta1 < 0.0f || beta1 >= 1.0f ||
        !std::isfinite(beta2) ||
        beta2 < 0.0f || beta2 >= 1.0f ||
        !std::isfinite(epsilon) ||
        epsilon <= 0.0f) {
        return false;
    }

    if (!load_vector(
            input,
            first_moment_,
            first_moment_.size()) ||
        !load_vector(
            input,
            second_moment_,
            second_moment_.size())) {
        return false;
    }

    learning_rate_ = learning_rate;
    beta1_ = beta1;
    beta2_ = beta2;
    epsilon_ = epsilon;
    step_count_ =
        static_cast<std::size_t>(
            step_count);

    return true;
}
