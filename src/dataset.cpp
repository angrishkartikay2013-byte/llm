#include "dataset.hpp"

#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

TextDataset::TextDataset(
    const std::string& text,
    const Tokenizer& tokenizer)
    : tokens_(tokenizer.encode(text)) {}

const std::vector<int>& TextDataset::tokens() const {
    return tokens_;
}

std::size_t TextDataset::size() const {
    return tokens_.size();
}

std::vector<std::pair<std::vector<int>, int>>
TextDataset::next_token_examples(
    std::size_t max_context_length) const {

    std::vector<std::pair<std::vector<int>, int>> examples;

    if (tokens_.size() < 2 || max_context_length == 0) {
        return examples;
    }

    for (std::size_t position = 1;
         position < tokens_.size();
         ++position) {

        const std::size_t start =
            position > max_context_length
                ? position - max_context_length
                : 0;

        examples.emplace_back(
            std::vector<int>(
                tokens_.begin() +
                    static_cast<std::ptrdiff_t>(start),
                tokens_.begin() +
                    static_cast<std::ptrdiff_t>(position)),
            tokens_[position]);
    }

    return examples;
}

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Cannot open text file: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}
