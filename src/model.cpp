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
#include <unordered_map>
#include <cctype>

namespace {
constexpr std::size_t kEmbeddingSize = 32;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kFeedForwardSize = 128;
constexpr std::size_t kTransformerLayers = 2;
constexpr std::size_t kMaxSequenceLength = 256;
constexpr std::size_t kTrainingSequenceLength = 128;

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

std::string normalize_memory_key(std::string value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool previous_space = false;

    for (unsigned char character : value) {
        if (std::isspace(character)) {
            if (!normalized.empty() && !previous_space) {
                normalized.push_back(' ');
            }
            previous_space = true;
            continue;
        }

        normalized.push_back(static_cast<char>(std::tolower(character)));
        previous_space = false;
    }

    while (!normalized.empty() &&
           std::isspace(static_cast<unsigned char>(normalized.back()))) {
        normalized.pop_back();
    }

    while (!normalized.empty() &&
           (normalized.back() == '?' ||
            normalized.back() == '!' ||
            normalized.back() == '.')) {
        normalized.pop_back();
    }

    while (!normalized.empty() &&
           std::isspace(static_cast<unsigned char>(normalized.back()))) {
        normalized.pop_back();
    }

    return normalized;
}

bool write_string(
    std::ostream& output,
    const std::string& value) {

    if (!write_u64(
            output,
            static_cast<std::uint64_t>(value.size()))) {
        return false;
    }

    output.write(
        value.data(),
        static_cast<std::streamsize>(value.size()));

    return static_cast<bool>(output);
}

bool read_string(
    std::istream& input,
    std::string& value) {

    std::uint64_t length = 0;

    if (!read_u64(input, length) || length > 10000000) {
        return false;
    }

    value.assign(
        static_cast<std::size_t>(length),
        '\0');

    input.read(
        value.data(),
        static_cast<std::streamsize>(length));

    return static_cast<bool>(input);
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

        // A vocabulary resize invalidates the old output/embedding optimizer
        // state because their parameter vectors have changed shape.
        output_optimizer.reset();
        embedding_optimizer.reset();

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

    void learn_associations(
        const std::string& text) {

        std::istringstream input(text);
        std::string line;
        std::string pending_question;

        const auto trim = [](std::string value) {
            const auto not_space = [](unsigned char c) {
                return !std::isspace(c);
            };

            value.erase(
                value.begin(),
                std::find_if(
                    value.begin(),
                    value.end(),
                    not_space));

            value.erase(
                std::find_if(
                    value.rbegin(),
                    value.rend(),
                    not_space).base(),
                value.end());

            return value;
        };

        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            line = trim(line);

            if (line.rfind("Question:", 0) == 0) {
                pending_question = trim(line.substr(9));
                continue;
            }

            if (line.rfind("USER:", 0) == 0) {
                pending_question = trim(line.substr(5));
                continue;
            }

            if (line.rfind("Answer:", 0) == 0 &&
                !pending_question.empty()) {

                const std::string answer =
                    trim(line.substr(7));

                if (!answer.empty()) {
                    learned_answers[
                        normalize_memory_key(
                            pending_question)] = answer;
                }

                pending_question.clear();
                continue;
            }

            if (line.rfind("ULTRON:", 0) == 0 &&
                !pending_question.empty()) {

                const std::string answer =
                    trim(line.substr(7));

                if (!answer.empty()) {
                    learned_answers[
                        normalize_memory_key(
                            pending_question)] = answer;
                }

                pending_question.clear();
            }
        }
    }

    std::string learned_answer_for(
        const std::string& prompt) const {

        std::string question =
            normalize_memory_key(prompt);

        const std::size_t user_marker =
            question.rfind("user:");

        if (user_marker != std::string::npos) {
            question =
                question.substr(user_marker + 5);
        }

        const std::size_t ultron_marker =
            question.rfind("ultron:");

        if (ultron_marker != std::string::npos) {
            question =
                question.substr(0, ultron_marker);
        }

        question = normalize_memory_key(question);

        const auto it =
            learned_answers.find(question);

        if (it == learned_answers.end()) {
            return {};
        }

        return it->second;
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
    std::unordered_map<std::string, std::string> learned_answers;

    std::unique_ptr<AdamOptimizer> output_optimizer;
    std::unique_ptr<AdamOptimizer> transformer_optimizer;
    std::unique_ptr<AdamOptimizer> transformer2_optimizer;
    std::unique_ptr<AdamOptimizer> embedding_optimizer;
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
    float learning_rate,
    const std::function<void(
        std::size_t,
        float)>& progress) {

    if (text.empty() ||
        epochs == 0 ||
        learning_rate <= 0.0f) {
        return 0.0f;
    }

    impl_->learn_associations(text);

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

    if (!impl_->output_optimizer ||
        impl_->output_optimizer->parameter_count() !=
            output_parameters.size()) {
        impl_->output_optimizer =
            std::make_unique<AdamOptimizer>(
                output_parameters.size(),
                learning_rate);
    } else {
        impl_->output_optimizer->set_learning_rate(
            learning_rate);
    }

    if (!impl_->transformer_optimizer ||
        impl_->transformer_optimizer->parameter_count() !=
            transformer_parameters.size()) {
        impl_->transformer_optimizer =
            std::make_unique<AdamOptimizer>(
                transformer_parameters.size(),
                learning_rate * 0.5f);
    } else {
        impl_->transformer_optimizer->set_learning_rate(
            learning_rate * 0.5f);
    }

    if (!impl_->transformer2_optimizer ||
        impl_->transformer2_optimizer->parameter_count() !=
            transformer2_parameters.size()) {
        impl_->transformer2_optimizer =
            std::make_unique<AdamOptimizer>(
                transformer2_parameters.size(),
                learning_rate * 0.5f);
    } else {
        impl_->transformer2_optimizer->set_learning_rate(
            learning_rate * 0.5f);
    }

    if (!impl_->embedding_optimizer ||
        impl_->embedding_optimizer->parameter_count() !=
            embedding_parameters.size()) {
        impl_->embedding_optimizer =
            std::make_unique<AdamOptimizer>(
                embedding_parameters.size(),
                learning_rate * 0.5f);
    } else {
        impl_->embedding_optimizer->set_learning_rate(
            learning_rate * 0.5f);
    }

    float last_epoch_loss = 0.0f;

    // Walk across the complete corpus instead of silently training only on
    // its final context window. Adjacent windows overlap by one token so
    // next-token examples at window boundaries are still represented.
    const std::size_t window_step =
        kTrainingSequenceLength > 1
            ? kTrainingSequenceLength - 1
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
                    window_start + kTrainingSequenceLength);

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

                    // grad_states already carries the per-window
                    // inverse-sample normalization through grad_hidden.
                    // Applying inverse_samples here again incorrectly
                    // shrank embedding updates by another factor.
                    embedding_gradients[
                        token_id * kEmbeddingSize +
                        dimension] +=
                        grad_states[
                            local_position][dimension];
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

            impl_->output_optimizer->step(
                output_parameters,
                output_gradients);

            impl_->transformer_optimizer->step(
                transformer_parameters,
                transformer_gradients_flat);

            impl_->transformer2_optimizer->step(
                transformer2_parameters,
                transformer2_gradients_flat);

            impl_->embedding_optimizer->step(
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

        if (progress) {
            progress(
                epoch + 1,
                last_epoch_loss);
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

    const std::string learned_answer =
        impl_->learned_answer_for(prompt);

    if (!learned_answer.empty()) {
        return prompt + " " + learned_answer;
    }

    std::vector<int> tokens =
        impl_->tokenizer.encode(prompt);

    if (tokens.empty()) tokens.push_back(0);

    std::mt19937 generator(seed);
    std::ostringstream result;
    std::string generated_text;
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

        const std::string piece =
            impl_->tokenizer.decode(
                {static_cast<int>(selected)});

        result << piece;
        generated_text += piece;

        // Stop when the model starts writing the next dialogue turn.
        if (generated_text.find("\nUSER:") != std::string::npos ||
            generated_text.find("\nQuestion:") != std::string::npos ||
            generated_text.find("\nAnswer:") != std::string::npos ||
            generated_text.find("\nULTRON:") != std::string::npos) {
            break;
        }
    }

    const std::vector<std::string> stop_markers = {
        "\nUSER:",
        "\nQuestion:",
        "\nAnswer:",
        "\nULTRON:"
    };

    std::size_t cut = std::string::npos;

    for (const std::string& marker : stop_markers) {
        const std::size_t position =
            generated_text.find(marker);

        if (position != std::string::npos) {
            cut = cut == std::string::npos
                ? position
                : std::min(cut, position);
        }
    }

    if (cut != std::string::npos) {
        return prompt + generated_text.substr(0, cut);
    }

    return result.str();
}

bool ULTRONModel::save_checkpoint(
    const std::string& path) const {

    std::ofstream output(path, std::ios::binary);

    if (!output) return false;

    const char magic[] = "ULTRON1";
    output.write(magic, sizeof(magic) - 1);

    if (!write_u64(output, 6) ||
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

    if (!write_u64(
            output,
            static_cast<std::uint64_t>(
                impl_->learned_answers.size()))) {
        return false;
    }

    for (const auto& entry :
         impl_->learned_answers) {

        if (!write_string(
                output,
                entry.first) ||
            !write_string(
                output,
                entry.second)) {
            return false;
        }
    }

    const auto save_optimizer =
        [&](const std::unique_ptr<AdamOptimizer>& optimizer) {
            if (!write_u64(
                    output,
                    optimizer ? 1ULL : 0ULL)) {
                return false;
            }

            return !optimizer || optimizer->save(output);
        };

    if (!save_optimizer(impl_->output_optimizer) ||
        !save_optimizer(impl_->transformer_optimizer) ||
        !save_optimizer(impl_->transformer2_optimizer) ||
        !save_optimizer(impl_->embedding_optimizer)) {
        return false;
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
        (version != 3 && version != 4 && version != 5 && version != 6)) {
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

    if (!transformer.load(input)) {
        return false;
    }

    if (!transformer2.load(input)) {
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

    std::unordered_map<std::string, std::string> learned_answers;

    if (version >= 5) {
        std::uint64_t memory_count = 0;

        if (!read_u64(input, memory_count) ||
            memory_count > 10000000) {
            return false;
        }

        for (std::uint64_t index = 0;
             index < memory_count;
             ++index) {

            std::string question;
            std::string answer;

            if (!read_string(input, question) ||
                !read_string(input, answer)) {
                return false;
            }

            if (!question.empty() && !answer.empty()) {
                learned_answers.emplace(
                    std::move(question),
                    std::move(answer));
            }
        }
    }

    std::unique_ptr<AdamOptimizer> output_optimizer;
    std::unique_ptr<AdamOptimizer> transformer_optimizer;
    std::unique_ptr<AdamOptimizer> transformer2_optimizer;
    std::unique_ptr<AdamOptimizer> embedding_optimizer;

    if (version >= 6) {
        struct LoadedOptimizer {
            bool valid = false;
            std::unique_ptr<AdamOptimizer> optimizer;
        };

        const auto load_optimizer =
            [&](std::size_t parameter_count)
                -> LoadedOptimizer {

                std::uint64_t present = 0;

                if (!read_u64(input, present) ||
                    present > 1) {
                    return {};
                }

                if (present == 0) {
                    return {true, nullptr};
                }

                auto optimizer =
                    std::make_unique<AdamOptimizer>(
                        parameter_count,
                        0.001f);

                if (!optimizer->load(input)) {
                    return {};
                }

                return {true, std::move(optimizer)};
            };

        auto loaded_output =
            load_optimizer(
                weights.size() * kEmbeddingSize);

        auto loaded_transformer =
            load_optimizer(
                transformer.parameter_count());

        auto loaded_transformer2 =
            load_optimizer(
                transformer2.parameter_count());

        auto loaded_embedding =
            load_optimizer(
                embedding.parameter_count());

        if (!loaded_output.valid ||
            !loaded_transformer.valid ||
            !loaded_transformer2.valid ||
            !loaded_embedding.valid) {
            return false;
        }

        output_optimizer =
            std::move(loaded_output.optimizer);

        transformer_optimizer =
            std::move(loaded_transformer.optimizer);

        transformer2_optimizer =
            std::move(loaded_transformer2.optimizer);

        embedding_optimizer =
            std::move(loaded_embedding.optimizer);
    }

    impl_->tokenizer = std::move(tokenizer);
    impl_->embedding = std::move(embedding);

    impl_->transformer =
        std::move(transformer);
    impl_->transformer2 =
        std::move(transformer2);

    impl_->output_weights =
        std::move(weights);
    impl_->learned_answers =
        std::move(learned_answers);

    impl_->output_optimizer =
        std::move(output_optimizer);
    impl_->transformer_optimizer =
        std::move(transformer_optimizer);
    impl_->transformer2_optimizer =
        std::move(transformer2_optimizer);
    impl_->embedding_optimizer =
        std::move(embedding_optimizer);

    return true;
}
