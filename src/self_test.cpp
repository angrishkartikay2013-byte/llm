#include "self_test.hpp"

#include "model.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "transformer.hpp"
#include "tokenizer.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
void assert_close(
    double left,
    double right,
    double tolerance) {
    assert(std::isfinite(left));
    assert(std::isfinite(right));
    assert(std::fabs(left - right) <= tolerance);
}
}

int run_ultron_smoke_tests() {
    // Tensor math.
    Tensor a(2, 2);
    Tensor b(2, 2);
    a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f; a.at(1, 1) = 4.0f;
    b.at(0, 0) = 5.0f; b.at(0, 1) = 6.0f;
    b.at(1, 0) = 7.0f; b.at(1, 1) = 8.0f;

    const Tensor product = a.matmul(b);
    assert(std::fabs(product.at(0, 0) - 19.0f) < 1e-5f);
    assert(std::fabs(product.at(1, 1) - 50.0f) < 1e-5f);

    // Fresh tokenizer: byte fallback + learned BPE merges.
    const std::string tokenizer_text =
        "Hello, hello! Hello, ULTRON. "
        "How are you? How are you? "
        "I'm ready. I'm ready. ";

    Tokenizer tokenizer;
    tokenizer.train(tokenizer_text);

    assert(tokenizer.vocabulary_size() > 257);

    const std::string unseen_text =
        "NeverSeenWord42?!\nI'm-ready.";

    const auto encoded =
        tokenizer.encode(unseen_text);

    assert(!encoded.empty());

    for (const int token : encoded) {
        assert(token > 0);
        assert(
            static_cast<std::size_t>(token) <
            tokenizer.vocabulary_size());
    }

    assert(
        tokenizer.decode(encoded) ==
        unseen_text);

    // Tokenizer serialization must preserve the learned merge table.
    std::stringstream tokenizer_stream(
        std::ios::in |
        std::ios::out |
        std::ios::binary);

    assert(tokenizer.save(tokenizer_stream));

    Tokenizer restored_tokenizer;
    tokenizer_stream.seekg(0);
    assert(restored_tokenizer.load(tokenizer_stream));

    assert(
        restored_tokenizer.decode(
            restored_tokenizer.encode(unseen_text)) ==
        unseen_text);

    // Deterministic tokenizer training is important for reproducible models.
    Tokenizer tokenizer_again;
    tokenizer_again.train(tokenizer_text);

    assert(
        tokenizer_again.vocabulary_size() ==
        tokenizer.vocabulary_size());

    assert(
        tokenizer_again.encode(unseen_text) ==
        tokenizer.encode(unseen_text));

    // Adam sanity and checkpointable state.
    AdamOptimizer optimizer(2, 0.01f);
    std::vector<float> weight{1.0f, -1.0f};
    optimizer.step(
        weight,
        std::vector<float>{0.5f, -0.25f});

    assert(weight[0] < 1.0f);
    assert(weight[1] > -1.0f);
    assert(optimizer.step_count() == 1);

    std::stringstream optimizer_stream(
        std::ios::in |
        std::ios::out |
        std::ios::binary);

    assert(optimizer.save(optimizer_stream));

    AdamOptimizer restored_optimizer(2, 0.01f);
    optimizer_stream.seekg(0);
    assert(restored_optimizer.load(optimizer_stream));
    assert(restored_optimizer.step_count() == 1);
    assert(
        std::fabs(
            restored_optimizer.learning_rate() -
            optimizer.learning_rate()) <
        1e-8f);

    // Transformer forward/backward and finite gradients.
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

    // Train, evaluate, and verify learned-memory behavior.
    const std::string corpus =
        "hello ultron hello ultron hello ultron "
        "the system learns language from repeated examples. "
        "good morning good morning good morning. "
        "how are you how are you how are you.";

    ULTRONModel continuous;
    continuous.train(
        corpus,
        2,
        0.001f);

    const auto continuous_metrics =
        continuous.evaluate(corpus);

    assert(continuous_metrics.samples > 0);
    assert(std::isfinite(continuous_metrics.mean_loss));
    assert(std::isfinite(continuous_metrics.perplexity));
    assert(std::isfinite(continuous_metrics.accuracy));

    continuous.train(
        "Question: what is the test answer?\n"
        "Answer: learned memory works.",
        2,
        0.001f);

    const std::string generated =
        continuous.generate(
            "hello",
            4,
            0.2f,
            1,
            42);

    assert(!generated.empty());

    const std::string learned =
        continuous.generate(
            "USER: What is the test answer?\nULTRON:",
            8,
            0.2f,
            3,
            42);

    assert(
        learned.find(
            "learned memory works.") !=
        std::string::npos);

    // Critical regression test:
    // two uninterrupted epochs must match one epoch + checkpoint + one epoch.
    const std::string checkpoint =
        "ultron_smoke_checkpoint.bin";

    // A direct two-epoch run and a split run must follow the same optimizer
    // trajectory once the checkpoint contains tokenizer + weights + Adam state.
    ULTRONModel uninterrupted;
    uninterrupted.train(
        corpus,
        2,
        0.001f);

    ULTRONModel checkpointed;
    checkpointed.train(
        corpus,
        1,
        0.001f);

    const std::string continuation_checkpoint =
        "ultron_continuation_checkpoint.bin";

    assert(
        checkpointed.save_checkpoint(
            continuation_checkpoint));

    ULTRONModel continuation_restored;
    assert(
        continuation_restored.load_checkpoint(
            continuation_checkpoint));

    continuation_restored.train(
        corpus,
        1,
        0.001f);

    const auto uninterrupted_metrics =
        uninterrupted.evaluate(corpus);

    const auto continuation_metrics =
        continuation_restored.evaluate(corpus);

    assert_close(
        uninterrupted_metrics.mean_loss,
        continuation_metrics.mean_loss,
        1e-5);

    assert_close(
        uninterrupted_metrics.accuracy,
        continuation_metrics.accuracy,
        1e-8);

    const std::string restored_text =
        continuation_restored.generate(
            "hello",
            8,
            0.2f,
            1,
            42);

    const std::string uninterrupted_text =
        uninterrupted.generate(
            "hello",
            8,
            0.2f,
            1,
            42);

    assert(restored_text == uninterrupted_text);

    std::remove(
        checkpoint.c_str());
    std::remove(
        continuation_checkpoint.c_str());

    std::cout
        << "ULTRON smoke tests passed."
        << std::endl;

    return 0;
}
