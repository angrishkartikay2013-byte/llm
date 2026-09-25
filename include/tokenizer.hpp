#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    Tokenizer();

    void train(const std::string& text);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    std::size_t vocabulary_size() const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
};