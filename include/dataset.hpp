#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class Tokenizer;

class TextDataset {
public:
    TextDataset(
        const std::string& text,
        const Tokenizer& tokenizer);

    const std::vector<int>& tokens() const;
    std::size_t size() const;

    std::vector<std::pair<std::vector<int>, int>>
    next_token_examples(
        std::size_t max_context_length) const;

private:
    std::vector<int> tokens_;
};

std::string read_text_file(const std::string& path);
