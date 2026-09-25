#include "tensor.hpp"

#include <stdexcept>

Tensor::Tensor()
    : rows_(0), cols_(0), data_() {}

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[row * cols_ + col];
}

const float& Tensor::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[row * cols_ + col];
}

std::size_t Tensor::rows() const {
    return rows_;
}

std::size_t Tensor::cols() const {
    return cols_;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Tensor shape mismatch for matmul");
    }

    Tensor result(rows_, other.cols_);

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const float a = at(i, k);
            for (std::size_t j = 0; j < other.cols_; ++j) {
                result.at(i, j) += a * other.at(k, j);
            }
        }
    }

    return result;
}

Tensor Tensor::add(const Tensor& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Tensor shape mismatch for add");
    }

    Tensor result(rows_, cols_);

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result.at(i, j) = at(i, j) + other.at(i, j);
        }
    }

    return result;
}
