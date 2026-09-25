#include "tokenizer.hpp"

#include <sstream>
#include <stdexcept>

Tokenizer::Tokenizer()
    : token_to_id_(), id_to_token_() {
    token_to_id_["<unk>"] = 0;
    id_to_token_.push_back("<unk>");
}

void Tokenizer::train(const std::string& text) {
    std::istringstream stream(text);
    std::string token;

    while (stream >> token) {
        if (token_to_id_.find(token) == token_to_id_.end()) {
            const int id = static_cast<int>(id_to_token_.size());
            token_to_id_[token] = id;
            id_to_token_.push_back(token);
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::istringstream stream(text);
    std::vector<int> tokens;
    std::string token;

    while (stream >> token) {
        const auto it = token_to_id_.find(token);
        tokens.push_back(
            it == token_to_id_.end() ? 0 : it->second
        );
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const int id = tokens[i];
        const std::string& token =
            (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size())
                ? id_to_token_[static_cast<std::size_t>(id)]
                : id_to_token_[0];

        if (!result.empty()) {
            result += ' ';
        }
        result += token;
    }

    return result;
}

std::size_t Tokenizer::vocabulary_size() const {
    return id_to_token_.size();
}
