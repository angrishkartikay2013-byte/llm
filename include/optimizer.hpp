#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

class AdamOptimizer {
public:
    explicit AdamOptimizer(
        std::size_t parameter_count,
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f);

    void step(
        std::vector<float>& weights,
        const std::vector<float>& gradients);

    void set_learning_rate(float learning_rate);

    float learning_rate() const;
    std::size_t parameter_count() const;
    std::size_t step_count() const;

    bool save(std::ostream& output) const;
    bool load(std::istream& input);

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    std::size_t step_count_;
    std::vector<float> first_moment_;
    std::vector<float> second_moment_;
};
