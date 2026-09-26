#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
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
    using Merge = std::pair<int, int>;

    static constexpr std::uint64_t kFormatMagic =
        0x554c54524f4e4252ULL; // "ULTRONBR"
    static constexpr std::uint64_t kFormatVersion = 3;
    static constexpr std::size_t kByteVocabularySize = 256;
    static constexpr std::size_t kMaxMerges = 512;

    static std::vector<std::string> legacy_split(
        const std::string& text,
        bool preserve_layout);

    void initialize_bpe_base();
    void learn_bpe(const std::string& text);
    static std::vector<std::string> split_bpe_units(const std::string& text);
    void apply_merges(std::vector<int>& tokens) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<Merge> merges_;

    // New checkpoints use byte-level BPE. Legacy checkpoints keep the old
    // word-tokenizer mode so they remain readable.
    bool bpe_mode_ = true;
    bool bpe_trained_ = false;
    bool bpe_boundary_mode_ = true;
    bool legacy_preserve_layout_ = true;
};