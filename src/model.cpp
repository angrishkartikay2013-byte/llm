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
#include <limits>
#include <random>
#include <sstream>
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

bool write_u64(std::ostream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(output);
}

bool read_u64(std::istream& input, std::uint64_t& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}
}

class ULTRONModel::Impl {
public:
    Impl()
        : tokenizer(),
          embedding(1, kEmbeddingSize),
          transformer(kEmbeddingSize, kHeads, kFeedForwardSize),
          output_weights(),
          positional(kMaxSequenceLength,
                     std::vector<float>(kEmbeddingSize, 0.0f)) {
        tokenizer.train(starter_corpus);
        rebuild_trainable_parameters();
        initialize_positions();
    }

    void rebuild_trainable_parameters() {
        const std::size_t vocabulary = tokenizer.vocabulary_size();
        embedding = Embedding(vocabulary, kEmbeddingSize);
        output_weights.assign(
            vocabulary,
            std::vector<float>(kEmbeddingSize, 0.0f));

        std::mt19937 generator(91);
        const float limit =
            std::sqrt(6.0f / static_cast<float>(
                vocabulary + kEmbeddingSize));
        std::uniform_real_distribution<float> distribution(
            -limit, limit);

        for (auto& row : output_weights) {
            for (float& value : row) {
                value = distribution(generator);
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
                        std::pow(10000.0f, exponent));
            }
        }
    }

    std::vector<std::vector<float>> encode_context(
        const std::vector<int>& tokens) const {
        if (tokens.empty()) return {};

        const std::size_t start =
            tokens.size() > kMaxSequenceLength
                ? tokens.size() - kMaxSequenceLength
                : 0;

        std::vector<std::vector<float>> states;
        states.reserve(tokens.size() - start);

        for (std::size_t index = start;
             index < tokens.size();
             ++index) {
            const std::size_t token_id =
                tokens[index] < 0
                    ? 0
                    : static_cast<std::size_t>(tokens[index]);

            auto state = embedding.lookup(token_id);
            const std::size_t position = index - start;

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
    std::vector<std::vector<float>> output_weights;
    std::vector<std::vector<float>> positional;
};

ULTRONModel::ULTRONModel()
    : impl_(std::make_unique<Impl>()) {}

ULTRONModel::~ULTRONModel() = default;
ULTRONModel::ULTRONModel(ULTRONModel&&) noexcept = default;
ULTRONModel& ULTRONModel::operator=(ULTRONModel&&) noexcept = default;

float ULTRONModel::train(
    const std::string& text,
    std::size_t epochs,
    float learning_rate) {

    if (text.empty() || epochs == 0) return 0.0f;

    const std::size_t old_vocab =
        impl_->tokenizer.vocabulary_size();

    impl_->tokenizer.train(text);

    if (impl_->tokenizer.vocabulary_size() != old_vocab) {
        impl_->rebuild_trainable_parameters();
    }

    const auto tokens = impl_->tokenizer.encode(text);

    if (tokens.size() < 2) return 0.0f;

    const std::size_t output_parameter_count =
        impl_->output_weights.size() * kEmbeddingSize;

    AdamOptimizer output_optimizer(
        output_parameter_count,
        learning_rate);

    AdamOptimizer embedding_optimizer(
        kEmbeddingSize,
        learning_rate * 0.5f);

    std::vector<float> weights(output_parameter_count);
    std::vector<float> gradients(output_parameter_count);

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

    double total_loss = 0.0;
    std::size_t samples = 0;

    for (std::size_t epoch = 0;
         epoch < epochs;
         ++epoch) {

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

            const auto& hidden =
                hidden_states.back();

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
                        weights[
                            token * kEmbeddingSize +
                            dimension];
                }
            }

            const auto probabilities =
                ultron_softmax(current_logits);

            const std::size_t target =
                static_cast<std::size_t>(
                    std::max(
                        tokens[position + 1],
                        0));

            total_loss +=
                ultron_cross_entropy_loss(
                    probabilities,
                    target);

            ++samples;

            ultron_output_gradient(
                hidden,
                probabilities,
                target,
                gradients);

            std::vector<float> hidden_gradient(
                kEmbeddingSize,
                0.0f);

            for (std::size_t token = 0;
                 token < probabilities.size();
                 ++token) {

                const float error =
                    probabilities[token] -
                    (token == target ? 1.0f : 0.0f);

                for (std::size_t dimension = 0;
                     dimension < kEmbeddingSize;
                     ++dimension) {
                    hidden_gradient[dimension] +=
                        error *
                        weights[
                            token * kEmbeddingSize +
                            dimension];
                }
            }

            const std::size_t input_token =
                static_cast<std::size_t>(
                    std::max(tokens[position], 0));

            embedding_optimizer.step(
                impl_->embedding.lookup_mutable(
                    input_token),
                hidden_gradient);

            output_optimizer.step(
                weights,
                gradients);
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

    return samples == 0
        ? 0.0f
        : static_cast<float>(
            total_loss /
            static_cast<double>(samples));
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
            impl_->logits(hidden_states.back());

        std::vector<float> scaled_logits(
            raw_logits.size());

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

        if (top_k > 0 &&
            top_k < candidates.size()) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() +
                    static_cast<std::ptrdiff_t>(top_k),
                candidates.end(),
                [&](std::size_t a, std::size_t b) {
                    return scaled_logits[a] >
                           scaled_logits[b];
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

bool ULTRONModel::save_checkpoint(
    const std::string& path) const {

    std::ofstream output(path, std::ios::binary);
    if (!output) return false;

    const char magic[] = "ULTRON1";
    output.write(magic, sizeof(magic) - 1);

    if (!write_u64(output, 1) ||
        !impl_->tokenizer.save(output) ||
        !impl_->embedding.save(output)) {
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
            reinterpret_cast<const char*>(row.data()),
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
        std::string(
            magic,
            sizeof(magic)) !=
            std::string(
                expected,
                sizeof(expected) - 1)) {
        return false;
    }

    std::uint64_t version = 0;

    if (!read_u64(input, version) ||
        version != 1) {
        return false;
    }

    Tokenizer tokenizer;

    if (!tokenizer.load(input)) return false;

    Embedding embedding(1, kEmbeddingSize);

    if (!embedding.load(input) ||
        embedding.embedding_size() != kEmbeddingSize ||
        embedding.vocabulary_size() !=
            tokenizer.vocabulary_size()) {
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
    impl_->output_weights = std::move(weights);

    return true;
}
