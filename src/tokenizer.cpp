#include "tokenizer.hpp"

#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>
#include <utility>

#include <string>

namespace {
bool write_u64(std::ostream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(output);
}
bool read_u64(std::istream& input, std::uint64_t& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}
}

Tokenizer::Tokenizer() { token_to_id_["<unk>"] = 0; id_to_token_.push_back("<unk>"); }

void Tokenizer::train(const std::string& text) {
    std::istringstream stream(text); std::string token;
    while (stream >> token) {
        if (token_to_id_.find(token) == token_to_id_.end()) {
            const int id = static_cast<int>(id_to_token_.size());
            token_to_id_[token] = id; id_to_token_.push_back(token);
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::istringstream stream(text); std::vector<int> tokens; std::string token;
    while (stream >> token) {
        const auto it = token_to_id_.find(token);
        tokens.push_back(it == token_to_id_.end() ? 0 : it->second);
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (const int id : tokens) {
        const std::string& token = (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size()) ? id_to_token_[static_cast<std::size_t>(id)] : id_to_token_[0];
        if (!result.empty()) result += ' ';
        result += token;
    }
    return result;
}

std::size_t Tokenizer::vocabulary_size() const { return id_to_token_.size(); }

bool Tokenizer::save(std::ostream& output) const {
    if (!write_u64(output, static_cast<std::uint64_t>(id_to_token_.size()))) return false;
    for (const std::string& token : id_to_token_) {
        if (!write_u64(output, static_cast<std::uint64_t>(token.size()))) return false;
        output.write(token.data(), static_cast<std::streamsize>(token.size()));
        if (!output) return false;
    }
    return true;
}

bool Tokenizer::load(std::istream& input) {
    std::uint64_t count = 0;
    if (!read_u64(input, count) || count == 0 || count > 10000000) return false;
    std::vector<std::string> loaded;
    loaded.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t length = 0;
        if (!read_u64(input, length) || length > 10000000) return false;
        std::string token(static_cast<std::size_t>(length), '\0');
        input.read(token.data(), static_cast<std::streamsize>(length));
        if (!input) return false;
        loaded.push_back(std::move(token));
    }
    token_to_id_.clear(); id_to_token_.clear();
    for (std::size_t i = 0; i < loaded.size(); ++i) {
        token_to_id_[loaded[i]] = static_cast<int>(i);
        id_to_token_.push_back(std::move(loaded[i]));
    }
    return !id_to_token_.empty();
}
