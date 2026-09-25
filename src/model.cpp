#include "model.hpp"

#include "embedding.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

#include <algorithm>
#include <cmath>
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

        embedding = Embedding(
            tokenizer.vocabulary_size(),
            kEmbeddingSize);

        output_weights.resize(
            tokenizer.vocabulary_size(),
            std::vector<float>(kEmbeddingSize, 0.0f));

        std::mt19937 generator(91);
        const float limit =
            std::sqrt(6.0f / static_cast<float>(
                tokenizer.vocabulary_size() + kEmbeddingSize));
        std::uniform_real_distribution<float> distribution(
            -limit, limit);

        for (auto& row : output_weights) {
            for (float& value : row) {
                value = distribution(generator);
            }
        }

        for (std::size_t position = 0;
             position < kMaxSequenceLength; ++position) {
            for (std::size_t dimension = 0;
                 dimension < kEmbeddingSize; ++dimension) {
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

void ULTRONModel::train(const std::string& text) {
    impl_->tokenizer.train(text);
}

std::string ULTRONModel::generate(
    const std::string& prompt) const {

    std::vector<int> tokens = impl_->tokenizer.encode(prompt);

    if (tokens.empty()) {
        tokens.push_back(0);
    }

    if (tokens.size() > kMaxSequenceLength) {
        tokens.erase(
            tokens.begin(),
            tokens.end() - static_cast<std::ptrdiff_t>(
                kMaxSequenceLength));
    }

    std::vector<std::vector<float>> states;
    states.reserve(tokens.size());

    for (std::size_t position = 0; position < tokens.size(); ++position) {
        auto state = impl_->embedding.lookup(
            static_cast<std::size_t>(std::max(tokens[position], 0)));

        for (std::size_t d = 0; d < kEmbeddingSize; ++d) {
            state[d] += impl_->positional[position][d];
        }

        states.push_back(std::move(state));
    }

    const auto hidden = impl_->transformer.forward(states);
    const auto& last = hidden.back();

    std::size_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();

    for (std::size_t token = 0;
         token < impl_->output_weights.size();
         ++token) {

        float logit = 0.0f;
        for (std::size_t d = 0; d < kEmbeddingSize; ++d) {
            logit += last[d] * impl_->output_weights[token][d];
        }

        if (logit > best_logit) {
            best_logit = logit;
            best_token = token;
        }
    }

    const std::string next = impl_->tokenizer.decode(
        {static_cast<int>(best_token)});

    if (next.empty()) {
        return "hello";
    }

    return next;
}
