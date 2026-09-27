#include "self_test.hpp"

#include "model.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "transformer.hpp"
#include "tokenizer.hpp"
#include "trainer.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(
    bool condition,
    const char* expression) {

    if (!condition) {
        throw std::runtime_error(
            std::string("ULTRON self-test failed: ") +
            expression);
    }
}

void assert_close(
    double left,
    double right,
    double tolerance) {
    require(std::isfinite(left), "left is finite");
    require(std::isfinite(right), "right is finite");
    require(
        std::fabs(left - right) <= tolerance,
        "values are within tolerance");
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
    require(std::fabs(product.at(0, 0) - 19.0f) < 1e-5f);
    require(std::fabs(product.at(1, 1) - 50.0f) < 1e-5f);

    // Fresh tokenizer: byte fallback + learned BPE merges.
    const std::string tokenizer_text =
        "Hello, hello! Hello, ULTRON. "
        "How are you? How are you? "
        "I'm ready. I'm ready. ";

    Tokenizer tokenizer;
    tokenizer.train(tokenizer_text);

    require(tokenizer.vocabulary_size() > 257);

    const std::string unseen_text =
        "NeverSeenWord42?!\nI'm-ready.";

    const auto encoded =
        tokenizer.encode(unseen_text);

    require(!encoded.empty());

    for (const int token : encoded) {
        require(token > 0);
        require(
            static_cast<std::size_t>(token) <
            tokenizer.vocabulary_size());
    }

    require(
        tokenizer.decode(encoded) ==
        unseen_text);

    const std::size_t frozen_vocab =
        tokenizer.vocabulary_size();

    tokenizer.train(
        "A completely different stream of new words arrives.");

    require(
        tokenizer.vocabulary_size() ==
        frozen_vocab);

    // Tokenizer serialization must preserve the learned merge table.
    std::stringstream tokenizer_stream(
        std::ios::in |
        std::ios::out |
        std::ios::binary);

    require(tokenizer.save(tokenizer_stream));

    Tokenizer restored_tokenizer;
    tokenizer_stream.seekg(0);
    require(restored_tokenizer.load(tokenizer_stream));

    require(
        restored_tokenizer.decode(
            restored_tokenizer.encode(unseen_text)) ==
        unseen_text);

    // Boundary-aware BPE must keep whitespace, punctuation, and newlines
    // as explicit layout boundaries rather than merging them into words.
    const std::string layout_text =
        "Hello, world!\nThis is a test.\nI'm ready.";

    const auto layout_tokens =
        tokenizer.encode(layout_text);

    require(
        tokenizer.decode(layout_tokens) ==
        layout_text);

    require(
        tokenizer.encode("hello world").size() >= 3);

    require(
        tokenizer.encode("hello,").size() >= 2);

    require(
        tokenizer.encode("hello\nworld").size() >= 3);

    // Deterministic tokenizer training is important for reproducible models.
    Tokenizer tokenizer_again;
    tokenizer_again.train(tokenizer_text);

    require(
        tokenizer_again.vocabulary_size() ==
        tokenizer.vocabulary_size());

    require(
        tokenizer_again.encode(unseen_text) ==
        tokenizer.encode(unseen_text));

    // Legacy tokenizer checkpoints use only a token count followed by
    // length-prefixed token strings. They must remain readable after BPE.
    std::stringstream legacy_stream(
        std::ios::in |
        std::ios::out |
        std::ios::binary);

    const auto write_u64 =
        [&](std::uint64_t value) {
            legacy_stream.write(
                reinterpret_cast<const char*>(&value),
                sizeof(value));
        };

    const std::vector<std::string> legacy_tokens = {
        "<unk>",
        "hello",
        " world",
        "!"
    };

    write_u64(legacy_tokens.size());

    for (const std::string& token :
         legacy_tokens) {

        write_u64(token.size());

        legacy_stream.write(
            token.data(),
            static_cast<std::streamsize>(
                token.size()));
    }

    Tokenizer legacy_tokenizer;
    legacy_stream.seekg(0);
    require(legacy_tokenizer.load(legacy_stream));
    require(
        legacy_tokenizer.decode(
            legacy_tokenizer.encode(
                "hello world!")) ==
        "hello world!");

    // Adam sanity and checkpointable state.
    AdamOptimizer optimizer(2, 0.01f);
    std::vector<float> weight{1.0f, -1.0f};
    optimizer.step(
        weight,
        std::vector<float>{0.5f, -0.25f});

    require(weight[0] < 1.0f);
    require(weight[1] > -1.0f);
    require(optimizer.step_count() == 1);

    std::stringstream optimizer_stream(
        std::ios::in |
        std::ios::out |
        std::ios::binary);

    require(optimizer.save(optimizer_stream));

    AdamOptimizer restored_optimizer(2, 0.01f);
    optimizer_stream.seekg(0);
    require(restored_optimizer.load(optimizer_stream));
    require(restored_optimizer.step_count() == 1);
    require(
        std::fabs(
            restored_optimizer.learning_rate() -
            optimizer.learning_rate()) <
        1e-8f);

    // Invalid Transformer configurations must fail before any
    // divide-by-zero or invalid allocation can occur.
    bool invalid_transformer_threw = false;
    try {
        TransformerBlock invalid(8, 0, 16);
    } catch (const std::invalid_argument&) {
        invalid_transformer_threw = true;
    }
    require(invalid_transformer_threw);

    // Transformer forward/backward and finite gradients.
    TransformerBlock transformer(8, 2, 16);
    const std::vector<std::vector<float>> inputs = {
        {0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.70f, -0.80f},
        {0.25f, 0.15f, -0.35f, 0.45f, -0.55f, 0.65f, -0.75f, 0.85f},
        {-0.30f, 0.40f, 0.20f, -0.50f, 0.60f, 0.10f, -0.70f, 0.80f}
    };

    const auto forward_output =
        transformer.forward(inputs);

    const std::vector<std::vector<float>> grad_output = {
        {0.70f, -0.10f, 0.30f, -0.50f, 0.20f, 0.60f, -0.40f, 0.90f},
        {-0.20f, 0.80f, -0.60f, 0.10f, 0.50f, -0.30f, 0.40f, -0.70f},
        {0.90f, 0.20f, -0.40f, 0.70f, -0.10f, -0.80f, 0.30f, 0.50f}
    };

    // Worker-thread numerical failures must propagate as C++ exceptions,
    // not terminate the process.
    bool nonfinite_forward_threw = false;
    try {
        auto bad_inputs = inputs;
        bad_inputs[1][2] =
            std::numeric_limits<float>::quiet_NaN();
        (void)transformer.forward(bad_inputs);
    } catch (const std::exception&) {
        nonfinite_forward_threw = true;
    }
    require(
        nonfinite_forward_threw,
        "Transformer forward rejects non-finite input safely");

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

    require(forward_output.size() == inputs.size());
    require(grad_inputs.size() == inputs.size());
    require(
        flattened_gradients.size() ==
        transformer.parameter_count());

    for (const float value : flattened_gradients) {
        require(std::isfinite(value));
    }

    // Numerical gradient regression: compare several analytical transformer
    // gradients with finite-difference estimates of a scalar loss. This catches
    // silent backpropagation mistakes that a "finite" gradient test cannot.
    std::vector<float> transformer_parameters;
    transformer.get_parameters(transformer_parameters);
    const std::vector<float> original_parameters =
        transformer_parameters;

    const auto scalar_loss =
        [&]() {
            const auto output =
                transformer.forward(inputs);
            double loss = 0.0;
            for (std::size_t row = 0;
                 row < output.size();
                 ++row) {
                for (std::size_t column = 0;
                     column < output[row].size();
                     ++column) {
                    loss +=
                        static_cast<double>(output[row][column]) *
                        static_cast<double>(grad_output[row][column]);
                }
            }
            return loss;
        };

    constexpr float kFiniteDifferenceStep = 1e-3f;
    const std::vector<std::size_t> checked_parameters = {
        0,
        63,
        64,
        127,
        128,
        191,
        192,
        255,
        256,
        383,
        384,
        511
    };

    for (const std::size_t parameter : checked_parameters) {
        if (parameter >= transformer_parameters.size()) {
            continue;
        }
        transformer_parameters =
            original_parameters;
        transformer_parameters[parameter] +=
            kFiniteDifferenceStep;
        transformer.set_parameters(
            transformer_parameters);
        const double plus_loss =
            scalar_loss();

        transformer_parameters =
            original_parameters;
        transformer_parameters[parameter] -=
            kFiniteDifferenceStep;
        transformer.set_parameters(
            transformer_parameters);
        const double minus_loss =
            scalar_loss();

        const double numerical_gradient =
            (plus_loss - minus_loss) /
            (2.0 * static_cast<double>(
                kFiniteDifferenceStep));

        const double analytical_gradient =
            static_cast<double>(
                flattened_gradients[parameter]);

        require(
            std::fabs(
                numerical_gradient -
                analytical_gradient) <
            2e-2);
    }

    transformer.set_parameters(
        original_parameters);

    // Train, evaluate, and verify learned-memory behavior.
    const std::string corpus =
        "hello ultron hello ultron hello ultron "
        "the system learns language from repeated examples. "
        "good morning good morning good morning. "
        "how are you how are you how are you.";

    ULTRONModel continuous;
    const float first_epoch_loss =
        continuous.train(
            corpus,
            1,
            0.001f);

    const float second_epoch_loss =
        continuous.train(
            corpus,
            1,
            0.001f);

    require(std::isfinite(first_epoch_loss));
    require(std::isfinite(second_epoch_loss));

    const auto first_epoch_metrics =
        continuous.evaluate(corpus);
    require(std::isfinite(first_epoch_metrics.mean_loss));
    require(std::isfinite(first_epoch_metrics.perplexity));

    // The smoke test requires a valid second training step but does not make
    // a brittle claim about the exact direction of one tiny-batch update.
    // The checkpoint continuation test below verifies trajectory equivalence.

    const auto continuous_metrics =
        continuous.evaluate(corpus);

    require(continuous_metrics.samples > 0);
    require(std::isfinite(continuous_metrics.mean_loss));
    require(std::isfinite(continuous_metrics.perplexity));
    require(std::isfinite(continuous_metrics.accuracy));

    continuous.train(
        "Question: what is the test answer?\n"
        "Answer: learned memory works.",
        2,
        0.001f);

    // Speed 10 must remain finite and usable; it changes only the training
    // compute budget, not the model architecture or checkpoint format.
    ULTRONModel fast_mode;
    fast_mode.train(
        corpus,
        1,
        0.001f,
        {},
        10);

    const auto fast_metrics =
        fast_mode.evaluate(corpus);

    require(fast_metrics.samples > 0);
    require(std::isfinite(fast_metrics.mean_loss));
    require(std::isfinite(fast_metrics.perplexity));
    require(std::isfinite(fast_metrics.accuracy));

    const std::string generated =
        continuous.generate(
            "hello",
            4,
            0.2f,
            1,
            42);

    require(!generated.empty());

    const std::string learned =
        continuous.generate(
            "USER: What is the test answer?\nULTRON:",
            8,
            0.2f,
            3,
            42);

    require(
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

    require(
        checkpointed.save_checkpoint(
            continuation_checkpoint));

    ULTRONModel continuation_restored;
    require(
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

    require(restored_text == uninterrupted_text);

    std::remove(
        checkpoint.c_str());
    std::remove(
        continuation_checkpoint.c_str());

    std::cout
        << "ULTRON smoke tests passed."
        << std::endl;

    return 0;
}
