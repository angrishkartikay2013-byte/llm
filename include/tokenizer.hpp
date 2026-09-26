#pragma once

#include <cstddef>
#include <iosfwd>
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

    bool save(std::ostream& output) const;
    bool load(std::istream& input);

private:
    static std::vector<std::string> split(
        const std::string& text,
        bool preserve_layout);

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    bool preserve_layout_ = true;
};
