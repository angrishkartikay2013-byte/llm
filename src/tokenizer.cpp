#include "tokenizer.hpp"

#include <cctype>
#include <cstdint>
#include <istream>
#include <ostream>
#include <utility>

namespace {
bool write_u64(std::ostream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(output);
}

bool read_u64(std::istream& input, std::uint64_t& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}

bool is_punctuation(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' ||
           c == ':' || c == ';' || c == '(' || c == ')' ||
           c == '[' || c == ']' || c == '{' || c == '}' ||
           c == '"' || c == '\'' || c == '-' || c == '/' ||
           c == '\\' || c == '+' || c == '=' || c == '*' ||
           c == '&' || c == '%' || c == '#' || c == '@';
}

bool is_word_character(char c) {
    return std::isalnum(
               static_cast<unsigned char>(c)) != 0 ||
           c == '_';
}
}

Tokenizer::Tokenizer() {
    token_to_id_["<unk>"] = 0;
    id_to_token_.push_back("<unk>");
}

std::vector<std::string> Tokenizer::split(
    const std::string& text,
    bool preserve_layout) {

    std::vector<std::string> tokens;
    std::string current;
    std::string pending_whitespace;

    auto flush_word = [&]() {
        if (current.empty()) {
            return;
        }

        std::string token;

        if (preserve_layout) {
            token = pending_whitespace;
        }

        for (const char c : current) {
            token += static_cast<char>(
                std::tolower(
                    static_cast<unsigned char>(c)));
        }

        tokens.push_back(std::move(token));
        current.clear();
        pending_whitespace.clear();
    };

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];

        if (c == '\r' || c == '\n' || c == '\t' || c == ' ') {
            flush_word();

            if (!preserve_layout) {
                continue;
            }

            if (c == '\r') {
                if (i + 1 < text.size() && text[i + 1] == '\n') {
                    continue;
                }
                if (pending_whitespace.empty() ||
                    pending_whitespace.back() != '\n') {
                    pending_whitespace.push_back('\n');
                }
            } else if (c == '\n') {
                if (pending_whitespace.empty() ||
                    pending_whitespace.back() != '\n') {
                    pending_whitespace.push_back('\n');
                }
            } else {
                if (pending_whitespace.empty() ||
                    (pending_whitespace.back() != ' ' &&
                     pending_whitespace.back() != '\n')) {
                    pending_whitespace.push_back(' ');
                }
            }

            continue;
        }

        const bool apostrophe_inside =
            c == '\'' &&
            i > 0 &&
            i + 1 < text.size() &&
            is_word_character(text[i - 1]) &&
            is_word_character(text[i + 1]);

        const bool hyphen_inside =
            c == '-' &&
            i > 0 &&
            i + 1 < text.size() &&
            is_word_character(text[i - 1]) &&
            is_word_character(text[i + 1]);

        if (is_punctuation(c) &&
            !apostrophe_inside &&
            !hyphen_inside) {

            flush_word();

            std::string punctuation;

            const bool keep_space_before =
                preserve_layout &&
                !pending_whitespace.empty() &&
                (c == '(' || c == '[' || c == '{' ||
                 c == '"' || c == '\\'');

            if (keep_space_before) {
                punctuation = pending_whitespace;
            }

            punctuation += c;
            tokens.push_back(std::move(punctuation));
            pending_whitespace.clear();
            continue;
        }

        current += c;
    }

    flush_word();

    return tokens;
}

void Tokenizer::train(const std::string& text) {
    for (const std::string& token :
         split(text, preserve_layout_)) {

        if (token_to_id_.find(token) == token_to_id_.end()) {
            const int id =
                static_cast<int>(id_to_token_.size());

            token_to_id_[token] = id;
            id_to_token_.push_back(token);
        }
    }
}

std::vector<int> Tokenizer::encode(
    const std::string& text) const {

    std::vector<int> tokens;

    for (const std::string& token :
         split(text, preserve_layout_)) {

        const auto it = token_to_id_.find(token);

        tokens.push_back(
            it == token_to_id_.end()
                ? 0
                : it->second);
    }

    return tokens;
}

std::string Tokenizer::decode(
    const std::vector<int>& tokens) const {

    std::string result;

    for (const int id : tokens) {
        const std::string& token =
            (id >= 0 &&
             static_cast<std::size_t>(id) < id_to_token_.size())
                ? id_to_token_[static_cast<std::size_t>(id)]
                : id_to_token_[0];

        if (preserve_layout_) {
            result += token;
            continue;
        }

        const bool punctuation =
            token.size() == 1 &&
            is_punctuation(token[0]);

        if (!result.empty() && !punctuation) {
            result += ' ';
        }

        result += token;
    }

    return result;
}

std::size_t Tokenizer::vocabulary_size() const {
    return id_to_token_.size();
}

bool Tokenizer::save(std::ostream& output) const {
    if (!write_u64(
            output,
            static_cast<std::uint64_t>(
                id_to_token_.size()))) {
        return false;
    }

    for (const std::string& token : id_to_token_) {
        if (!write_u64(
                output,
                static_cast<std::uint64_t>(
                    token.size()))) {
            return false;
        }

        output.write(
            token.data(),
            static_cast<std::streamsize>(
                token.size()));

        if (!output) {
            return false;
        }
    }

    return true;
}

bool Tokenizer::load(std::istream& input) {
    std::uint64_t count = 0;

    if (!read_u64(input, count) ||
        count == 0 ||
        count > 10000000) {
        return false;
    }

    std::vector<std::string> loaded;
    loaded.reserve(static_cast<std::size_t>(count));

    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t length = 0;

        if (!read_u64(input, length) ||
            length > 10000000) {
            return false;
        }

        std::string token(
            static_cast<std::size_t>(length),
            '\0');

        input.read(
            token.data(),
            static_cast<std::streamsize>(length));

        if (!input) {
            return false;
        }

        loaded.push_back(std::move(token));
    }

    token_to_id_.clear();
    id_to_token_.clear();

    for (std::size_t i = 0; i < loaded.size(); ++i) {
        token_to_id_[loaded[i]] =
            static_cast<int>(i);

        id_to_token_.push_back(
            std::move(loaded[i]));
    }

    preserve_layout_ = false;

    for (std::size_t i = 1; i < id_to_token_.size(); ++i) {
        const std::string& token = id_to_token_[i];

        if (!token.empty() &&
            (token.front() == ' ' ||
             token.front() == '\n')) {
            preserve_layout_ = true;
            break;
        }
    }

    return !id_to_token_.empty();
}
