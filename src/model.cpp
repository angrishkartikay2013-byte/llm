#include "model.hpp"

#include "embedding.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include "trainer.hpp"
#include "transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kEmbeddingSize = 32;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kFeedForwardSize = 128;
constexpr std::size_t kTransformerLayers = 2;
constexpr std::size_t kMaxSequenceLength = 128;

const char* starter_corpus =
    "hello i am ultron "
    "ultron is a language model "
    "ultron learns language from data "
    "this is a model built from scratch "
    "language models predict the next token "
    "transformers use attention to process context";

bool write_u64(std::ostream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(output);
}

bool read_u64(std::istream& input, std::uint64_t& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}

void scale_in_place(
    std::vector<float>& values,
    float scale) {

    for (float& value : values) {
        value *= scale;
    }
}

void clip_gradients(
    std::vector<float>& gradients,
    float max_norm) {

    if (gradients.empty() || max_norm <= 0.0f) {
        return;
    }

    double squared_norm = 0.0;

    for (const float value : gradients) {
        squared_norm +=
            static_cast<double>(value) *
            static_cast<double>(value);
    }

    const double norm =
        std::sqrt(squared_norm);

    if (norm <= static_cast<double>(max_norm) ||
        norm <= 1e-12) {
        return;
    }

    const float scale =
        max_norm /
        static_cast<float>(norm);

    scale_in_place(
        gradients,
        scale);
}
}

class ULTRONModel::Impl {
public:
    Impl()
        : tokenizer(),
          embedding(1, kEmbeddingSize),
          transformer(kEmbeddingSize, kHeads, kFeedForwardSize, 11),
          transformer2(kEmbeddingSize, kHeads, kFeedForwardSize, 101),
          output_weights(),
          positional(
              kMaxSequenceLength,
              std::vector<float>(
                  kEmbeddingSize,
                  0.0f)) {
        tokenizer.train(starter_corpus);
        rebuild_trainable_parameters();
        initialize_positions();
    }

    void rebuild_trainable_parameters() {
        const std::size_t vocabulary =
            tokenizer.vocabulary_size();

        const std::size_t old_vocabulary =
            embedding.vocabulary_size();

        if (vocabulary == old_vocabulary) {
            return;
        }

        std::vector<std::vector<float>> old_embeddings;
        old_embeddings.reserve(old_vocabulary);

        for (std::size_t token = 0;
             token < old_vocabulary;
             ++token) {
            old_embeddings.push_back(
                embedding.lookup(token));
        }

        const auto old_output_weights =
            output_weights;

        embedding =
            Embedding(
                vocabulary,
                kEmbeddingSize);

        output_weights.assign(
            vocabulary,
            std::vector<float>(
                kEmbeddingSize,
                0.0f));

        std::mt19937 generator(91);
        const float limit =
            std::sqrt(
                6.0f /
                static_cast<float>(
                    vocabulary +
                    kEmbeddingSize));

        std::uniform_real_distribution<float> distribution(
            -limit,
            limit);

        for (auto& row : output_weights) {
            for (float& value : row) {
                value = distribution(generator);
            }
        }

        const std::size_t preserved =
            std::min(old_vocabulary, vocabulary);

        for (std::size_t token = 0;
             token < preserved;
             ++token) {
            embedding.lookup_mutable(token) =
                old_embeddings[token];

            if (token < old_output_weights.size()) {
                output_weights[token] =
                    old_output_weights[token];
            }
        }
    }

    void initialize_positions() {
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
                        std::pow(
                            10000.0f,
                            exponent));
            }
        }
    }

    std::vector<std::vector<float>> make_context_states(
        const std::vector<int>& tokens,
        std::size_t& context_start) const {

        if (tokens.empty()) {
            context_start = 0;
            return {};
        }

        context_start =
            tokens.size() > kMaxSequenceLength
                ? tokens.size() - kMaxSequenceLength
                : 0;

        std::vector<std::vector<float>> states;
        states.reserve(
            tokens.size() - context_start);

        for (std::size_t index = context_start;
             index < tokens.size();
             ++index) {

            const std::size_t token_id =
                tokens[index] < 0
                    ? 0
                    : static_cast<std::size_t>(
                        tokens[index]);

            auto state =
                embedding.lookup(token_id);

            const std::size_t position =
                index - context_start;

            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize;
                 ++dimension) {
                state[dimension] +=
                    positional[position][dimension];
            }

            states.push_back(
                std::move(state));
        }

        return states;
    }

    std::vector<std::vector<float>> encode_context(
        const std::vector<int>& tokens) const {

        std::size_t context_start = 0;

        const auto states =
            make_context_states(
                tokens,
                context_start);

        (void)context_start;

        if (states.empty()) {
            return {};
        }

        auto hidden = transformer.forward(states);
        for (std::size_t layer = 1; layer < kTransformerLayers; ++layer) {
            hidden = transformer2.forward(hidden);
        }
        return hidden;
    }

    std::vector<float> logits(
        const std::vector<float>& hidden) const {

        std::vector<float> result(
            output_weights.size(),
            0.0f);

        for (std::size_t token = 0;
             token < output_weights.size();
             ++token) {

            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize;
                 ++dimension) {

                result[token] +=
                    hidden[dimension] *
                    output_weights[token][dimension];
            }
        }

        return result;
    }

    Tokenizer tokenizer;
    Embedding embedding;
    TransformerBlock transformer;
    TransformerBlock transformer2;
    std::vector<std::vector<float>> output_weights;
    std::vector<std::vector<float>> positional;
};

ULTRONModel::ULTRONModel()
    : impl_(std::make_unique<Impl>()) {}

ULTRONModel::~ULTRONModel() = default;
ULTRONModel::ULTRONModel(ULTRONModel&&) noexcept = default;

ULTRONModel& ULTRONModel::operator=(
    ULTRONModel&&) noexcept = default;

float ULTRONModel::train(
    const std::string& text,
    std::size_t epochs,
    float learning_rate) {

    if (text.empty() ||
        epochs == 0 ||
        learning_rate <= 0.0f) {
        return 0.0f;
    }

    const std::size_t old_vocab =
        impl_->tokenizer.vocabulary_size();

    impl_->tokenizer.train(text);

    if (impl_->tokenizer.vocabulary_size() != old_vocab) {
        impl_->rebuild_trainable_parameters();
    }

    const auto tokens =
        impl_->tokenizer.encode(text);

    if (tokens.size() < 2) {
        return 0.0f;
    }

    std::vector<float> output_parameters;
    for (const auto& row : impl_->output_weights) {
        output_parameters.insert(
            output_parameters.end(),
            row.begin(),
            row.end());
    }

    std::vector<float> transformer_parameters;
    impl_->transformer.get_parameters(
        transformer_parameters);

    std::vector<float> transformer2_parameters;
    impl_->transformer2.get_parameters(
        transformer2_parameters);

    std::vector<float> embedding_parameters;
    impl_->embedding.get_parameters(
        embedding_parameters);

    AdamOptimizer output_optimizer(
        output_parameters.size(),
        learning_rate);

    AdamOptimizer transformer_optimizer(
        transformer_parameters.size(),
        learning_rate * 0.5f);

    AdamOptimizer transformer2_optimizer(
        transformer2_parameters.size(),
        learning_rate * 0.5f);

    AdamOptimizer embedding_optimizer(
        embedding_parameters.size(),
        learning_rate * 0.5f);

    float last_epoch_loss = 0.0f;

    // Walk across the complete corpus instead of silently training only on
    // its final context window. Adjacent windows overlap by one token so
    // next-token examples at window boundaries are still represented.
    const std::size_t window_step =
        kMaxSequenceLength > 1
            ? kMaxSequenceLength - 1
            : 1;

    for (std::size_t epoch = 0;
         epoch < epochs;
         ++epoch) {

        double epoch_loss = 0.0;
        std::size_t epoch_samples = 0;

        for (std::size_t window_start = 0;
             window_start < tokens.size();
             window_start += window_step) {

            const std::size_t window_end =
                std::min(
                    tokens.size(),
                    window_start + kMaxSequenceLength);

            const std::size_t window_size =
                window_end - window_start;

            if (window_size < 2) {
                break;
            }

            std::vector<int> window_tokens(
                tokens.begin() +
                    static_cast<std::ptrdiff_t>(
                        window_start),
                tokens.begin() +
                    static_cast<std::ptrdiff_t>(
                        window_end));

            std::size_t context_start = 0;
            const auto states =
                impl_->make_context_states(
                    window_tokens,
                    context_start);

            const auto first_hidden_states =
                impl_->transformer.forward(states);
            const auto hidden_states =
                impl_->transformer2.forward(first_hidden_states);

            if (hidden_states.size() < 2) {
                continue;
            }

            std::vector<std::vector<float>> grad_hidden(
                hidden_states.size(),
                std::vector<float>(
                    kEmbeddingSize,
                    0.0f));

            std::vector<float> output_gradients(
                output_parameters.size(),
                0.0f);

            std::size_t window_samples = 0;

            for (std::size_t position = 0;
                 position + 1 < hidden_states.size();
                 ++position) {

                const auto& hidden =
                    hidden_states[position];

                std::vector<float> current_logits(
                    impl_->output_weights.size(),
                    0.0f);

                for (std::size_t token = 0;
                     token < impl_->output_weights.size();
                     ++token) {

                    for (std::size_t dimension = 0;
                         dimension < kEmbeddingSize;
                         ++dimension) {

                        current_logits[token] +=
                            hidden[dimension] *
                            output_parameters[
                                token * kEmbeddingSize +
                                dimension];
                    }
                }

                const auto probabilities =
                    ultron_softmax(
                        current_logits);

                const std::size_t target =
                    static_cast<std::size_t>(
                        std::max(
                            window_tokens[position + 1],
                            0));

                epoch_loss +=
                    ultron_cross_entropy_loss(
                        probabilities,
                        target);

                ++epoch_samples;
                ++window_samples;

                for (std::size_t token = 0;
                     token < probabilities.size();
                     ++token) {

                    const float error =
                        probabilities[token] -
                        (token == target ? 1.0f : 0.0f);

                    for (std::size_t dimension = 0;
                         dimension < kEmbeddingSize;
                         ++dimension) {

                        const std::size_t parameter =
                            token * kEmbeddingSize +
                            dimension;

                        output_gradients[parameter] +=
                            error *
                            hidden[dimension];

                        grad_hidden[position][dimension] +=
                            error *
                            output_parameters[
                                parameter];
                    }
                }
            }

            if (window_samples == 0) {
                continue;
            }

            const float inverse_samples =
                1.0f /
                static_cast<float>(
                    window_samples);

            scale_in_place(
                output_gradients,
                inverse_samples);

            for (auto& row : grad_hidden) {
                scale_in_place(
                    row,
                    inverse_samples);
            }

            TransformerBlock::Gradients transformer2_gradients;
            std::vector<std::vector<float>> grad_first_hidden;

            impl_->transformer2.backward(
                first_hidden_states,
                grad_hidden,
                grad_first_hidden,
                transformer2_gradients);

            std::vector<float> transformer2_gradients_flat;
            impl_->transformer2.flatten_gradients(
                transformer2_gradients,
                transformer2_gradients_flat);

            TransformerBlock::Gradients transformer_gradients;
            std::vector<std::vector<float>> grad_states;

            impl_->transformer.backward(
                states,
                grad_first_hidden,
                grad_states,
                transformer_gradients);

            std::vector<float> transformer_gradients_flat;
            impl_->transformer.flatten_gradients(
                transformer_gradients,
                transformer_gradients_flat);

            std::vector<float> embedding_gradients(
                embedding_parameters.size(),
                0.0f);

            for (std::size_t local_position = 0;
                 local_position < grad_states.size();
                 ++local_position) {

                const int token_value =
                    window_tokens[local_position];

                if (token_value < 0) {
                    continue;
                }

                const std::size_t token_id =
                    static_cast<std::size_t>(
                        token_value);

                if (token_id >=
                    impl_->embedding.vocabulary_size()) {
                    continue;
                }

                for (std::size_t dimension = 0;
                     dimension < kEmbeddingSize;
                     ++dimension) {

                    embedding_gradients[
                        token_id * kEmbeddingSize +
                        dimension] +=
                        grad_states[
                            local_position][dimension] *
                        inverse_samples;
                }
            }

            clip_gradients(
                output_gradients,
                5.0f);

            clip_gradients(
                transformer_gradients_flat,
                5.0f);

            clip_gradients(
                transformer2_gradients_flat,
                5.0f);

            clip_gradients(
                embedding_gradients,
                5.0f);

            output_optimizer.step(
                output_parameters,
                output_gradients);

            transformer_optimizer.step(
                transformer_parameters,
                transformer_gradients_flat);

            transformer2_optimizer.step(
                transformer2_parameters,
                transformer2_gradients_flat);

            embedding_optimizer.step(
                embedding_parameters,
                embedding_gradients);

            impl_->transformer.set_parameters(
                transformer_parameters);
            impl_->transformer2.set_parameters(
                transformer2_parameters);

            impl_->embedding.set_parameters(
                embedding_parameters);

            for (std::size_t token = 0;
                 token < impl_->output_weights.size();
                 ++token) {

                std::copy(
                    output_parameters.begin() +
                        static_cast<std::ptrdiff_t>(
                            token * kEmbeddingSize),
                    output_parameters.begin() +
                        static_cast<std::ptrdiff_t>(
                            (token + 1) * kEmbeddingSize),
                    impl_->output_weights[token].begin());
            }

            if (window_end == tokens.size()) {
                break;
            }
        }

        last_epoch_loss =
            epoch_samples == 0
                ? 0.0f
                : static_cast<float>(
                    epoch_loss /
                    static_cast<double>(
                        epoch_samples));

        if ((epoch + 1) == epochs ||
            (epoch + 1) % 5 == 0) {

            std::cout
                << "epoch "
                << (epoch + 1)
                << "/"
                << epochs
                << " loss="
                << last_epoch_loss
                << '\n';
        }
    }

    return last_epoch_loss;
}
ModelEvaluation ULTRONModel::evaluate(
    const std::string& text) const {

    ModelEvaluation result;

    if (text.empty()) {
        return result;
    }

    const auto tokens =
        impl_->tokenizer.encode(text);

    if (tokens.size() < 2) {
        return result;
    }

    double total_loss = 0.0;
    std::size_t correct = 0;

    for (std::size_t position = 0;
         position + 1 < tokens.size();
         ++position) {

        std::vector<int> context(
            tokens.begin(),
            tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    position + 1));

        const auto hidden_states =
            impl_->encode_context(context);

        if (hidden_states.empty()) continue;

        const auto logits =
            impl_->logits(
                hidden_states.back());

        const auto probabilities =
            ultron_softmax(logits);

        const std::size_t target =
            static_cast<std::size_t>(
                std::max(
                    tokens[position + 1],
                    0));

        if (target >= probabilities.size()) continue;

        total_loss +=
            ultron_cross_entropy_loss(
                probabilities,
                target);

        const std::size_t prediction =
            static_cast<std::size_t>(
                std::max_element(
                    probabilities.begin(),
                    probabilities.end()) -
                probabilities.begin());

        if (prediction == target) ++correct;

        ++result.samples;
    }

    if (result.samples == 0) {
        return result;
    }

    result.mean_loss =
        total_loss /
        static_cast<double>(result.samples);

    result.perplexity =
        std::exp(result.mean_loss);

    result.accuracy =
        static_cast<double>(correct) /
        static_cast<double>(result.samples);

    return result;
}

std::string ULTRONModel::generate(
    const std::string& prompt,
    std::size_t max_new_tokens,
    float temperature,
    std::size_t top_k,
    unsigned int seed) const {

    if (max_new_tokens == 0) return prompt;
    if (temperature <= 0.0f) temperature = 1.0f;

    std::vector<int> tokens =
        impl_->tokenizer.encode(prompt);

    if (tokens.empty()) tokens.push_back(0);

    std::mt19937 generator(seed);
    std::ostringstream result;
    result << prompt;

    for (std::size_t generated = 0;
         generated < max_new_tokens;
         ++generated) {

        const auto hidden_states =
            impl_->encode_context(tokens);

        if (hidden_states.empty()) break;

        const auto raw_logits =
            impl_->logits(
                hidden_states.back());

        std::vector<float> scaled_logits(
            raw_logits.size());

        for (std::size_t i = 0;
             i < raw_logits.size();
             ++i) {
            scaled_logits[i] =
                raw_logits[i] / temperature;
        }

        constexpr float repetition_penalty = 1.15f;

        for (const int previous_token : tokens) {
            if (previous_token < 0) continue;

            const std::size_t token_id =
                static_cast<std::size_t>(previous_token);

            if (token_id >= scaled_logits.size()) continue;

            if (scaled_logits[token_id] >= 0.0f) {
                scaled_logits[token_id] /=
                    repetition_penalty;
            } else {
                scaled_logits[token_id] *=
                    repetition_penalty;
            }
        }

        std::vector<std::size_t> candidates(
            scaled_logits.size());

        for (std::size_t i = 0;
             i < candidates.size();
             ++i) candidates[i] = i;

        if (top_k > 0 &&
            top_k < candidates.size()) {

            std::partial_sort(
                candidates.begin(),
                candidates.begin() +
                    static_cast<std::ptrdiff_t>(
                        top_k),
                candidates.end(),
                [&](std::size_t a, std::size_t b) {
                    return scaled_logits[a] >
                           scaled_logits[b];
                });

            candidates.resize(top_k);
        }

        std::vector<float> candidate_logits;
        candidate_logits.reserve(
            candidates.size());

        for (std::size_t candidate :
             candidates) {
            candidate_logits.push_back(
                scaled_logits[candidate]);
        }

        const auto probabilities =
            ultron_softmax(
                candidate_logits);

        std::discrete_distribution<std::size_t>
            sampler(
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

bool ULTRONModel::save_checkpoint(
    const std::string& path) const {

    std::ofstream output(path, std::ios::binary);

    if (!output) return false;

    const char magic[] = "ULTRON1";
    output.write(magic, sizeof(magic) - 1);

    if (!write_u64(output, 3) ||
        !impl_->tokenizer.save(output) ||
        !impl_->embedding.save(output) ||
        !impl_->transformer.save(output) ||
        !impl_->transformer2.save(output)) {
        return false;
    }

    if (!write_u64(
            output,
            static_cast<std::uint64_t>(
                impl_->output_weights.size())) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                kEmbeddingSize))) {
        return false;
    }

    for (const auto& row :
         impl_->output_weights) {

        output.write(
            reinterpret_cast<const char*>(
                row.data()),
            static_cast<std::streamsize>(
                row.size() * sizeof(float)));

        if (!output) return false;
    }

    return true;
}

bool ULTRONModel::load_checkpoint(
    const std::string& path) {

    std::ifstream input(path, std::ios::binary);

    if (!input) return false;

    const char expected[] = "ULTRON1";
    char magic[sizeof(expected) - 1]{};

    input.read(magic, sizeof(magic));

    if (!input ||
        std::string(magic, sizeof(magic)) !=
            std::string(
                expected,
                sizeof(expected) - 1)) {
        return false;
    }

    std::uint64_t version = 0;

    if (!read_u64(input, version) ||
        (version != 1 && version != 2 && version != 3)) {
        return false;
    }

    Tokenizer tokenizer;

    if (!tokenizer.load(input)) return false;

    Embedding embedding(
        1,
        kEmbeddingSize);

    if (!embedding.load(input) ||
        embedding.embedding_size() != kEmbeddingSize ||
        embedding.vocabulary_size() !=
            tokenizer.vocabulary_size()) {
        return false;
    }

    TransformerBlock transformer(
        kEmbeddingSize,
        kHeads,
        kFeedForwardSize,
        11);

    TransformerBlock transformer2(
        kEmbeddingSize,
        kHeads,
        kFeedForwardSize,
        101);

    if (version >= 2 &&
        !transformer.load(input)) {
        return false;
    }

    if (version >= 3 &&
        !transformer2.load(input)) {
        return false;
    }

    std::uint64_t vocabulary = 0;
    std::uint64_t dimension = 0;

    if (!read_u64(input, vocabulary) ||
        !read_u64(input, dimension) ||
        vocabulary != tokenizer.vocabulary_size() ||
        dimension != kEmbeddingSize) {
        return false;
    }

    std::vector<std::vector<float>> weights(
        static_cast<std::size_t>(vocabulary),
        std::vector<float>(
            static_cast<std::size_t>(dimension),
            0.0f));

    for (auto& row : weights) {
        input.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(
                row.size() * sizeof(float)));

        if (!input) return false;
    }

    impl_->tokenizer = std::move(tokenizer);
    impl_->embedding = std::move(embedding);

    if (version >= 2) {
        impl_->transformer =
            std::move(transformer);
    }

    if (version >= 3) {
        impl_->transformer2 =
            std::move(transformer2);
    }

    impl_->output_weights =
        std::move(weights);

    return true;
}
