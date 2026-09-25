#include "model.hpp"

#include "embedding.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include "trainer.hpp"
#include "transformer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kEmbeddingSize = 32;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kFeedForwardSize = 128;
constexpr std::size_t kMaxSequenceLength = 128;

const char* starter_corpus =
    "hello i am ultron "
    "ultron is a language model "
    "ultron learns language from data "
    "this is a model built from scratch "
    "language models predict the next token "
    "transformers use attention to process context";
}

class ULTRONModel::Impl {
public:
    Impl()
        : tokenizer(),
          embedding(1, kEmbeddingSize),
          transformer(kEmbeddingSize, kHeads, kFeedForwardSize),
          output_weights(),
          positional(
              kMaxSequenceLength,
              std::vector<float>(kEmbeddingSize, 0.0f)) {

        tokenizer.train(starter_corpus);
        rebuild_trainable_parameters();

        for (std::size_t position = 0;
             position < kMaxSequenceLength;
             ++position) {

            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize;
                 ++dimension) {

                const float exponent =
                    static_cast<float>(dimension) /
                    static_cast<float>(kEmbeddingSize);

                positional[position][dimension] =
                    std::sin(
                        static_cast<float>(position) /
                        std::pow(10000.0f, exponent));
            }
        }
    }

    void rebuild_trainable_parameters() {
        const std::size_t vocabulary =
            tokenizer.vocabulary_size();

        embedding = Embedding(vocabulary, kEmbeddingSize);

        output_weights.assign(
            vocabulary,
            std::vector<float>(kEmbeddingSize, 0.0f));

        std::mt19937 generator(91);
        const float limit =
            std::sqrt(
                6.0f / static_cast<float>(
                    vocabulary + kEmbeddingSize));

        std::uniform_real_distribution<float> distribution(
            -limit, limit);

        for (auto& row : output_weights) {
            for (float& value : row) {
                value = distribution(generator);
            }
        }
    }

    std::vector<std::vector<float>> encode_context(
        const std::vector<int>& tokens) const {

        std::vector<std::vector<float>> states;

        if (tokens.empty()) {
            return states;
        }

        const std::size_t start =
            tokens.size() > kMaxSequenceLength
                ? tokens.size() - kMaxSequenceLength
                : 0;

        states.reserve(tokens.size() - start);

        for (std::size_t index = start;
             index < tokens.size();
             ++index) {

            const std::size_t token_id =
                tokens[index] < 0
                    ? 0
                    : static_cast<std::size_t>(tokens[index]);

            auto state = embedding.lookup(token_id);

            const std::size_t position =
                index - start;

            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize;
                 ++dimension) {
                state[dimension] += positional[position][dimension];
            }

            states.push_back(std::move(state));
        }

        return transformer.forward(states);
    }

    std::vector<float> logits(
        const std::vector<float>& hidden) const {

        std::vector<float> result(
            output_weights.size(),
            0.0f);

        for (std::size_t token = 0;
             token < output_weights.size();
             ++token) {

            float value = 0.0f;

            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize;
                 ++dimension) {
                value +=
                    hidden[dimension] *
                    output_weights[token][dimension];
            }

            result[token] = value;
        }

        return result;
    }

    Tokenizer tokenizer;
    Embedding embedding;
    TransformerBlock transformer;
    std::vector<std::vector<float>> output_weights;
    std::vector<std::vector<float>> positional;
};

ULTRONModel::ULTRONModel()
    : impl_(std::make_unique<Impl>()) {}

ULTRONModel::~ULTRONModel() = default;

ULTRONModel::ULTRONModel(ULTRONModel&&) noexcept = default;

ULTRONModel& ULTRONModel::operator=(ULTRONModel&&) noexcept = default;

void ULTRONModel::train(
    const std::string& text,
    std::size_t epochs,
    float learning_rate) {

    if (text.empty() || epochs == 0) {
        return;
    }

    impl_->tokenizer.train(text);

    // Rebuild the output vocabulary so newly learned tokens get weights.
    impl_->rebuild_trainable_parameters();

    const auto tokens = impl_->tokenizer.encode(text);

    if (tokens.size() < 2) {
        return;
    }

    const std::size_t parameter_count =
        impl_->output_weights.size() * kEmbeddingSize;

    AdamOptimizer optimizer(
        parameter_count,
        learning_rate);

    std::vector<float> weights(parameter_count);
    std::vector<float> gradients(parameter_count);

    for (std::size_t token = 0;
         token < impl_->output_weights.size();
         ++token) {
        std::copy(
            impl_->output_weights[token].begin(),
            impl_->output_weights[token].end(),
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    token * kEmbeddingSize));
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t position = 0;
             position + 1 < tokens.size();
             ++position) {

            std::vector<int> context(
                tokens.begin(),
                tokens.begin() +
                    static_cast<std::ptrdiff_t>(position + 1));

            const auto hidden_states =
                impl_->encode_context(context);

            if (hidden_states.empty()) {
                continue;
            }

            const auto& hidden = hidden_states.back();

            std::vector<float> current_logits(
                impl_->output_weights.size(),
                0.0f);

            for (std::size_t token = 0;
                 token < impl_->output_weights.size();
                 ++token) {

                float value = 0.0f;

                for (std::size_t dimension = 0;
                     dimension < kEmbeddingSize;
                     ++dimension) {

                    value +=
                        hidden[dimension] *
                        weights[
                            token * kEmbeddingSize +
                            dimension];
                }

                current_logits[token] = value;
            }

            const auto probabilities =
                ultron_softmax(current_logits);

            ultron_output_gradient(
                hidden,
                probabilities,
                static_cast<std::size_t>(
                    std::max(tokens[position + 1], 0)),
                gradients);

            optimizer.step(weights, gradients);
        }
    }

    for (std::size_t token = 0;
         token < impl_->output_weights.size();
         ++token) {

        std::copy(
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    token * kEmbeddingSize),
            weights.begin() +
                static_cast<std::ptrdiff_t>(
                    (token + 1) * kEmbeddingSize),
            impl_->output_weights[token].begin());
    }
}

std::string ULTRONModel::generate(
    const std::string& prompt,
    std::size_t max_new_tokens,
    float temperature,
    std::size_t top_k,
    unsigned int seed) const {

    if (max_new_tokens == 0) {
        return prompt;
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    std::vector<int> tokens =
        impl_->tokenizer.encode(prompt);

    if (tokens.empty()) {
        tokens.push_back(0);
    }

    std::mt19937 generator(seed);
    std::ostringstream result;
    result << prompt;

    for (std::size_t generated = 0;
         generated < max_new_tokens;
         ++generated) {

        const auto hidden_states =
            impl_->encode_context(tokens);

        if (hidden_states.empty()) {
            break;
        }

        const auto raw_logits =
            impl_->logits(hidden_states.back());

        std::vector<float> scaled_logits(raw_logits.size());

        for (std::size_t i = 0;
             i < raw_logits.size();
             ++i) {
            scaled_logits[i] =
                raw_logits[i] / temperature;
        }

        std::vector<std::size_t> candidates(
            scaled_logits.size());

        for (std::size_t i = 0;
             i < candidates.size();
             ++i) {
            candidates[i] = i;
        }

        if (top_k > 0 && top_k < candidates.size()) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() +
                    static_cast<std::ptrdiff_t>(top_k),
                candidates.end(),
                [&](std::size_t left, std::size_t right) {
                    return scaled_logits[left] >
                           scaled_logits[right];
                });

            candidates.resize(top_k);
        }

        std::vector<float> candidate_logits;
        candidate_logits.reserve(candidates.size());

        for (std::size_t candidate : candidates) {
            candidate_logits.push_back(
                scaled_logits[candidate]);
        }

        const auto probabilities =
            ultron_softmax(candidate_logits);

        std::discrete_distribution<std::size_t> sampler(
            probabilities.begin(),
            probabilities.end());

        const std::size_t selected =
            candidates[sampler(generator)];

        tokens.push_back(
            static_cast<int>(selected));

        const std::string word =
            impl_->tokenizer.decode(
                {static_cast<int>(selected)});

        if (!word.empty()) {
            if (!result.str().empty() &&
                result.str().back() != ' ') {
                result << ' ';
            }
            result << word;
        }
    }

    return result.str();
}
