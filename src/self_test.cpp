#include "self_test.hpp"

#include "model.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int run_ultron_smoke_tests() {
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

    Tokenizer layout_tokenizer;
    const std::string layout_text =
        "Hello, I'm ready.\nHow are you?";
    layout_tokenizer.train(layout_text);

    const auto layout_tokens =
        layout_tokenizer.encode(layout_text);

    const std::string layout_decoded =
        layout_tokenizer.decode(layout_tokens);

    assert(
        layout_decoded ==
        "hello, i'm ready.\nhow are you?");

    AdamOptimizer optimizer(1, 0.01f);
    std::vector<float> weight{1.0f};
    optimizer.step(weight, std::vector<float>{0.5f});
    assert(weight[0] < 1.0f);

    TransformerBlock transformer(8, 2, 16);
    const std::vector<std::vector<float>> inputs(
        3,
        std::vector<float>(8, 0.1f));
    const auto forward_output =
        transformer.forward(inputs);

    std::vector<std::vector<float>> grad_output(
        3,
        std::vector<float>(8, 1.0f));
    std::vector<std::vector<float>> grad_inputs;
    TransformerBlock::Gradients transformer_gradients;

    transformer.backward(
        inputs,
        grad_output,
        grad_inputs,
        transformer_gradients);

    std::vector<float> flattened_gradients;
    transformer.flatten_gradients(
        transformer_gradients,
        flattened_gradients);

    assert(forward_output.size() == inputs.size());
    assert(grad_inputs.size() == inputs.size());
    assert(
        flattened_gradients.size() ==
        transformer.parameter_count());

    for (const float value : flattened_gradients) {
        assert(std::isfinite(value));
    }

    ULTRONModel model;
    model.train(
        "hello ultron hello ultron hello ultron",
        2,
        0.001f);

    const auto metrics =
        model.evaluate(
            "hello ultron hello ultron");

    assert(metrics.samples > 0);
    assert(std::isfinite(metrics.mean_loss));
    assert(std::isfinite(metrics.perplexity));

    model.train(
        "Question: what is the test answer?\nAnswer: learned memory works.",
        2,
        0.001f);

    const std::string generated =
        model.generate(
            "hello",
            4,
            0.2f,
            1,
            42);

    assert(!generated.empty());

    const std::string learned =
        model.generate(
            "USER: What is the test answer?\nULTRON:",
            8,
            0.2f,
            3,
            42);

    assert(
        learned.find("learned memory works.") !=
        std::string::npos);

    const std::string checkpoint =
        "ultron_smoke_checkpoint.bin";
    assert(model.save_checkpoint(checkpoint));

    ULTRONModel restored;
    assert(restored.load_checkpoint(checkpoint));

    const std::string restored_text =
        restored.generate(
            "hello",
            4,
            0.2f,
            1,
            42);

    assert(restored_text == generated);

    const std::string restored_learned =
        restored.generate(
            "USER: What is the test answer?\nULTRON:",
            8,
            0.2f,
            3,
            42);

    assert(
        restored_learned.find("learned memory works.") !=
        std::string::npos);

    std::remove(checkpoint.c_str());

    std::cout
        << "ULTRON smoke tests passed."
        << std::endl;

    return 0;
}
