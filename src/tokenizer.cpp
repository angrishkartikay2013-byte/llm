#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {
bool write_u64(
    std::ostream& output,
    std::uint64_t value) {

    output.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));

    return static_cast<bool>(output);
}

bool read_u64(
    std::istream& input,
    std::uint64_t& value) {

    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));

    return static_cast<bool>(input);
}

bool write_i32(
    std::ostream& output,
    std::int32_t value) {

    output.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));

    return static_cast<bool>(output);
}

bool read_i32(
    std::istream& input,
    std::int32_t& value) {

    input.read(
        reinterpret_cast<char*>(&value),
        sizeof(value));

    return static_cast<bool>(input);
}

std::uint64_t pair_key(
    int left,
    int right) {

    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(left))
            << 32U) |
           static_cast<std::uint32_t>(right);
}

bool is_legacy_punctuation(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' ||
           c == ':' || c == ';' || c == '(' || c == ')' ||
           c == '[' || c == ']' || c == '{' || c == '}' ||
           c == '"' || c == '\'' || c == '-' || c == '/' ||
           c == '\\' || c == '+' || c == '=' || c == '*' ||
           c == '&' || c == '%' || c == '#' || c == '@';
}

bool is_legacy_word_character(char c) {
    return std::isalnum(
               static_cast<unsigned char>(c)) != 0 ||
           c == '_';
}
}

Tokenizer::Tokenizer() {
    initialize_bpe_base();
}

void Tokenizer::initialize_bpe_base() {
    token_to_id_.clear();
    id_to_token_.clear();
    merges_.clear();

    // ID 0 is reserved as a genuine unknown token for malformed/legacy data.
    token_to_id_["<unk>"] = 0;
    id_to_token_.push_back("<unk>");

    // Every possible byte has a deterministic fallback token. This means
    // unseen words, symbols, punctuation, and UTF-8 bytes never become <unk>.
    for (std::size_t byte = 0;
         byte < kByteVocabularySize;
         ++byte) {

        std::string token(1, static_cast<char>(
            static_cast<unsigned char>(byte)));

        const int id =
            static_cast<int>(id_to_token_.size());

        token_to_id_[token] = id;
        id_to_token_.push_back(std::move(token));
    }

    bpe_mode_ = true;
    legacy_preserve_layout_ = true;
}

std::vector<std::string> Tokenizer::legacy_split(
    const std::string& text,
    bool preserve_layout) {

    std::vector<std::string> tokens;
    std::string current;
    std::string pending_whitespace;

    const auto flush_word = [&]() {
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

    for (std::size_t i = 0;
         i < text.size();
         ++i) {

        const char c = text[i];

        if (c == '\r' ||
            c == '\n' ||
            c == '\t' ||
            c == ' ') {

            flush_word();

            if (!preserve_layout) {
                continue;
            }

            if (c == '\r') {
                if (i + 1 < text.size() &&
                    text[i + 1] == '\n') {
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
            is_legacy_word_character(text[i - 1]) &&
            is_legacy_word_character(text[i + 1]);

        const bool hyphen_inside =
            c == '-' &&
            i > 0 &&
            i + 1 < text.size() &&
            is_legacy_word_character(text[i - 1]) &&
            is_legacy_word_character(text[i + 1]);

        if (is_legacy_punctuation(c) &&
            !apostrophe_inside &&
            !hyphen_inside) {

            flush_word();

            std::string punctuation;

            const bool keep_space_before =
                preserve_layout &&
                !pending_whitespace.empty() &&
                (c == '(' || c == '[' || c == '{' ||
                 c == '"' || c == '\'');

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

void Tokenizer::learn_bpe(
    const std::string& text) {

    merges_.clear();

    if (text.empty()) {
        return;
    }

    std::vector<std::vector<int>> sequences;
    sequences.emplace_back();
    sequences.back().reserve(text.size());

    // Do not learn merges across newline boundaries. Spaces may still be
    // merged with words (for example " the"), which preserves natural layout.
    for (const unsigned char byte :
         std::string(text.begin(), text.end())) {

        const int id =
            token_to_id_.at(
                std::string(
                    1,
                    static_cast<char>(byte)));

        sequences.back().push_back(id);

        if (byte == '\n' && !sequences.back().empty()) {
            sequences.emplace_back();
        }
    }

    if (!sequences.empty() &&
        sequences.back().empty()) {
        sequences.pop_back();
    }

    const std::size_t max_vocabulary =
        kByteVocabularySize +
        1 +
        kMaxMerges;

    while (merges_.size() < kMaxMerges &&
           id_to_token_.size() < max_vocabulary) {

        std::unordered_map<
            std::uint64_t,
            std::size_t> counts;

        for (const auto& sequence : sequences) {
            if (sequence.size() < 2) {
                continue;
            }

            for (std::size_t i = 1;
                 i < sequence.size();
                 ++i) {

                ++counts[
                    pair_key(
                        sequence[i - 1],
                        sequence[i])];
            }
        }

        if (counts.empty()) {
            break;
        }

        std::size_t best_count = 0;
        int best_left = std::numeric_limits<int>::max();
        int best_right = std::numeric_limits<int>::max();

        constexpr std::size_t kMaxTokenBytes = 16;

        for (const auto& entry : counts) {
            const int left =
                static_cast<int>(
                    static_cast<std::uint32_t>(
                        entry.first >> 32U));

            const int right =
                static_cast<int>(
                    static_cast<std::uint32_t>(
                        entry.first & 0xffffffffULL));

            if (left < 0 || right < 0 ||
                static_cast<std::size_t>(left) >= id_to_token_.size() ||
                static_cast<std::size_t>(right) >= id_to_token_.size()) {
                continue;
            }

            const std::size_t merged_bytes =
                id_to_token_[
                    static_cast<std::size_t>(left)].size() +
                id_to_token_[
                    static_cast<std::size_t>(right)].size();

            // Tiny models benefit more from reusable subwords than from
            // memorizing long, corpus-specific phrases.
            if (merged_bytes > kMaxTokenBytes) {
                continue;
            }

            const std::string merged_token =
                id_to_token_[
                    static_cast<std::size_t>(left)] +
                id_to_token_[
                    static_cast<std::size_t>(right)];

            if (token_to_id_.find(merged_token) !=
                token_to_id_.end()) {
                continue;
            }

            if (entry.second > best_count ||
                (entry.second == best_count &&
                 (left < best_left ||
                  (left == best_left &&
                   right < best_right)))) {

                best_count = entry.second;
                best_left = left;
                best_right = right;
            }
        }

        // Single occurrences do not justify creating a merge and would
        // inflate the vocabulary without improving compression.
        if (best_count < 2) {
            break;
        }

        const int new_id =
            static_cast<int>(
                id_to_token_.size());

        const std::string& left_token =
            id_to_token_[static_cast<std::size_t>(
                best_left)];

        const std::string& right_token =
            id_to_token_[static_cast<std::size_t>(
                best_right)];

        const std::string merged_token =
            left_token + right_token;

        token_to_id_[merged_token] = new_id;
        id_to_token_.push_back(
            merged_token);

        merges_.emplace_back(
            best_left,
            best_right);

        for (auto& sequence : sequences) {
            std::vector<int> merged;
            merged.reserve(sequence.size());

            std::size_t i = 0;

            while (i < sequence.size()) {
                if (i + 1 < sequence.size() &&
                    sequence[i] == best_left &&
                    sequence[i + 1] == best_right) {

                    merged.push_back(new_id);
                    i += 2;
                } else {
                    merged.push_back(sequence[i]);
                    ++i;
                }
            }

            sequence.swap(merged);
        }
    }
}

void Tokenizer::apply_merges(
    std::vector<int>& tokens) const {

    for (const Merge& merge : merges_) {
        if (tokens.size() < 2) {
            break;
        }

        std::vector<int> merged;
        merged.reserve(tokens.size());

        std::size_t index = 0;

        while (index < tokens.size()) {
            if (index + 1 < tokens.size() &&
                tokens[index] == merge.first &&
                tokens[index + 1] == merge.second) {

                const int merged_id =
                    static_cast<int>(
                        kByteVocabularySize + 1 +
                        (&merge - merges_.data()));

                merged.push_back(merged_id);
                index += 2;
            } else {
                merged.push_back(tokens[index]);
                ++index;
            }
        }

        tokens.swap(merged);
    }
}

void Tokenizer::train(
    const std::string& text) {

    if (bpe_mode_) {
        if (merges_.empty()) {
            learn_bpe(text);
        }
        return;
    }

    // Legacy checkpoints keep their original tokenizer semantics. This
    // preserves compatibility, while all fresh models use BPE.
    for (const std::string& token :
         legacy_split(text, legacy_preserve_layout_)) {

        if (token_to_id_.find(token) ==
            token_to_id_.end()) {

            const int id =
                static_cast<int>(
                    id_to_token_.size());

            token_to_id_[token] = id;
            id_to_token_.push_back(token);
        }
    }
}

std::vector<int> Tokenizer::encode(
    const std::string& text) const {

    if (!bpe_mode_) {
        std::vector<int> tokens;

        for (const std::string& token :
             legacy_split(
                 text,
                 legacy_preserve_layout_)) {

            const auto it =
                token_to_id_.find(token);

            tokens.push_back(
                it == token_to_id_.end()
                    ? 0
                    : it->second);
        }

        return tokens;
    }

    std::vector<int> tokens;
    tokens.reserve(text.size());

    for (const unsigned char byte :
         std::string(text.begin(), text.end())) {

        const auto it =
            token_to_id_.find(
                std::string(
                    1,
                    static_cast<char>(byte)));

        tokens.push_back(
            it == token_to_id_.end()
                ? 0
                : it->second);
    }

    apply_merges(tokens);
    return tokens;
}

std::string Tokenizer::decode(
    const std::vector<int>& tokens) const {

    std::string result;

    for (const int id : tokens) {
        if (id == 0) {
            result += "<unk>";
            continue;
        }

        if (id < 0 ||
            static_cast<std::size_t>(id) >=
                id_to_token_.size()) {
            result += "<unk>";
            continue;
        }

        result +=
            id_to_token_[static_cast<std::size_t>(id)];
    }

    return result;
}

std::size_t Tokenizer::vocabulary_size() const {
    return id_to_token_.size();
}

bool Tokenizer::save(
    std::ostream& output) const {

    if (!write_u64(
            output,
            kFormatMagic) ||
        !write_u64(
            output,
            kFormatVersion) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                bpe_mode_ ? 1 : 0)) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                legacy_preserve_layout_ ? 1 : 0)) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                id_to_token_.size())) ||
        !write_u64(
            output,
            static_cast<std::uint64_t>(
                merges_.size()))) {
        return false;
    }

    for (const std::string& token :
         id_to_token_) {

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

    for (const Merge& merge : merges_) {
        if (!write_i32(
                output,
                static_cast<std::int32_t>(
                    merge.first)) ||
            !write_i32(
                output,
                static_cast<std::int32_t>(
                    merge.second))) {
            return false;
        }
    }

    return true;
}

bool Tokenizer::load(
    std::istream& input) {

    std::uint64_t first = 0;

    if (!read_u64(input, first)) {
        return false;
    }

    if (first != kFormatMagic) {
        // Legacy tokenizer format: the first uint64 was simply the token
        // count. Reconstruct exactly the old word-level tokenizer state.
        const std::uint64_t count = first;

        if (count == 0 || count > 10000000) {
            return false;
        }

        std::vector<std::string> loaded;
        loaded.reserve(
            static_cast<std::size_t>(count));

        for (std::uint64_t i = 0;
             i < count;
             ++i) {

            std::uint64_t length = 0;

            if (!read_u64(
                    input,
                    length) ||
                length > 10000000) {
                return false;
            }

            std::string token(
                static_cast<std::size_t>(length),
                '\0');

            input.read(
                token.data(),
                static_cast<std::streamsize>(
                    length));

            if (!input) {
                return false;
            }

            loaded.push_back(
                std::move(token));
        }

        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();

        for (std::size_t i = 0;
             i < loaded.size();
             ++i) {

            token_to_id_[loaded[i]] =
                static_cast<int>(i);

            id_to_token_.push_back(
                std::move(loaded[i]));
        }

        legacy_preserve_layout_ = false;

        for (std::size_t i = 1;
             i < id_to_token_.size();
             ++i) {

            const std::string& token =
                id_to_token_[i];

            if (!token.empty() &&
                (token.front() == ' ' ||
                 token.front() == '\n')) {

                legacy_preserve_layout_ = true;
                break;
            }
        }

        bpe_mode_ = false;
        return !id_to_token_.empty();
    }

    std::uint64_t version = 0;
    std::uint64_t bpe_flag = 0;
    std::uint64_t layout_flag = 0;
    std::uint64_t count = 0;
    std::uint64_t merge_count = 0;

    if (!read_u64(input, version) ||
        version != kFormatVersion ||
        !read_u64(input, bpe_flag) ||
        !read_u64(input, layout_flag) ||
        !read_u64(input, count) ||
        !read_u64(input, merge_count)) {
        return false;
    }

    if (count == 0 ||
        count > 10000000 ||
        merge_count > kMaxMerges ||
        merge_count >= count) {
        return false;
    }

    std::vector<std::string> loaded;
    loaded.reserve(
        static_cast<std::size_t>(count));

    for (std::uint64_t i = 0;
         i < count;
         ++i) {

        std::uint64_t length = 0;

        if (!read_u64(
                input,
                length) ||
            length > 10000000) {
            return false;
        }

        std::string token(
            static_cast<std::size_t>(length),
            '\0');

        input.read(
            token.data(),
            static_cast<std::streamsize>(
                length));

        if (!input) {
            return false;
        }

        loaded.push_back(
            std::move(token));
    }

    std::vector<Merge> merges;
    merges.reserve(
        static_cast<std::size_t>(
            merge_count));

    for (std::uint64_t i = 0;
         i < merge_count;
         ++i) {

        std::int32_t left = 0;
        std::int32_t right = 0;

        if (!read_i32(input, left) ||
            !read_i32(input, right)) {
            return false;
        }

        if (left < 0 ||
            right < 0 ||
            static_cast<std::size_t>(left) >= loaded.size() ||
            static_cast<std::size_t>(right) >= loaded.size()) {
            return false;
        }

        const std::size_t merged_id =
            kByteVocabularySize + 1 +
            static_cast<std::size_t>(i);

        if (merged_id >= loaded.size() ||
            loaded[merged_id] !=
                loaded[static_cast<std::size_t>(left)] +
                loaded[static_cast<std::size_t>(right)]) {
            return false;
        }

        merges.emplace_back(
            static_cast<int>(left),
            static_cast<int>(right));
    }

    token_to_id_.clear();
    id_to_token_.clear();

    for (std::size_t i = 0;
         i < loaded.size();
         ++i) {

        if (token_to_id_.find(loaded[i]) !=
            token_to_id_.end()) {
            return false;
        }

        token_to_id_[loaded[i]] =
            static_cast<int>(i);

        id_to_token_.push_back(
            std::move(loaded[i]));
    }

    bpe_mode_ = bpe_flag != 0;
    legacy_preserve_layout_ =
        layout_flag != 0;
    merges_ = std::move(merges);

    if (bpe_mode_ &&
        id_to_token_.size() <
            kByteVocabularySize + 1) {
        return false;
    }

    return !id_to_token_.empty();
}
