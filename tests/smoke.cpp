#include "model.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    Tensor a(2, 2);
    Tensor b(2, 2);
    a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f; a.at(1, 1) = 4.0f;
    b.at(0, 0) = 5.0f; b.at(0, 1) = 6.0f;
    b.at(1, 0) = 7.0f; b.at(1, 1) = 8.0f;
    const Tensor product = a.matmul(b);
    assert(std::fabs(product.at(0, 0) - 19.0f) < 1e-5f);
    assert(std::fabs(product.at(1, 1) - 50.0f) < 1e-5f);

    Tokenizer tokenizer;
    tokenizer.train("hello world, hello ultron!");
    const auto encoded = tokenizer.encode("hello ultron!");
    assert(encoded.size() == 3);
    assert(tokenizer.vocabulary_size() >= 5);

    AdamOptimizer optimizer(1, 0.01f);
    std::vector<float> weight{1.0f};
    optimizer.step(weight, std::vector<float>{0.5f});
    assert(weight[0] < 1.0f);

    ULTRONModel model;
    model.train("hello ultron hello ultron hello ultron", 2, 0.01f);
    const auto metrics = model.evaluate("hello ultron hello ultron");
    assert(metrics.samples > 0);
    assert(std::isfinite(metrics.mean_loss));
    assert(std::isfinite(metrics.perplexity));
    const std::string generated = model.generate("hello", 4, 0.8f, 4, 42);
    assert(!generated.empty());

    std::cout << "ULTRON smoke tests passed." << std::endl;
    return 0;
}
