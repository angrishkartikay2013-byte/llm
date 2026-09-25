#pragma once

#include <cstddef>
#include <string>
#include <vector>

class TextDataset {
public:
    explicit TextDataset(std::string text);

    const std::vector<int>& tokens() const;
    std::size_t size() const;

    std::vector<std::pair<std::vector<int>, int>>
    next_token_examples(std::size_t max_context_length) const;

private:
    std::string text_;
    std::vector<int> tokens_;
};

std::string read_text_file(const std::string& path);
