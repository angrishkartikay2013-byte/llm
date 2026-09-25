#include "model.hpp"

#include "attention.hpp"
#include "embedding.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

#include <sstream>

namespace {
constexpr std::size_t kEmbeddingSize = 32;

const char* starter_corpus =
    "hello i am ultron "
    "ultron is a language model "
    "ultron learns language from data "
    "this is a model built from scratch";
}

ULTRONModel::ULTRONModel() = default;

void ULTRONModel::train(const std::string& text) {
    // Training infrastructure will be expanded in the trainer component.
    (void)text;
}

std::string ULTRONModel::generate(const std::string& prompt) const {
    // The first engine build keeps inference deterministic and local.
    // Learned next-token generation will be connected as the trainer evolves.
    if (prompt.empty()) {
        return "hello";
    }

    std::istringstream stream(prompt);
    std::string last_token;
    std::string token;

    while (stream >> token) {
        last_token = token;
    }

    if (last_token.empty()) {
        return "hello";
    }

    if (last_token == "hello") {
        return "i am ultron";
    }

    if (last_token == "i") {
        return "am ultron";
    }

    if (last_token == "ultron") {
        return "is learning";
    }

    return std::string("I received: ") + prompt;
}
