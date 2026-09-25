#pragma once

#include <cstddef>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(std::size_t rows, std::size_t cols);

    float& at(std::size_t row, std::size_t col);
    const float& at(std::size_t row, std::size_t col) const;

    std::size_t rows() const;
    std::size_t cols() const;

    Tensor matmul(const Tensor& other) const;
    Tensor add(const Tensor& other) const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};