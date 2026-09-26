#include "dictionary.hpp"

#include <cctype>
#include <cstdio>
#include <sstream>
#include <string>

namespace {

std::string url_encode_word(
    const std::string& word) {

    std::ostringstream encoded;

    constexpr char hex[] =
        "0123456789ABCDEF";

    for (const unsigned char character : word) {
        if (std::isalnum(character) ||
            character == '-' ||
            character == '_' ||
            character == '.') {
            encoded << static_cast<char>(character);
        } else {
            encoded
                << '%'
                << hex[(character >> 4) & 0x0F]
                << hex[character & 0x0F];
        }
    }

    return encoded.str();
}

std::string extract_json_string(
    const std::string& json,
    const std::string& key) {

    const std::string marker =
        """ + key + "":"";

    const std::size_t start =
        json.find(marker);

    if (start == std::string::npos) {
        return {};
    }

    std::string result;
    result.reserve(256);

    bool escaped = false;

    for (std::size_t index =
             start + marker.size();
         index < json.size();
         ++index) {

        const char character = json[index];

        if (escaped) {
            switch (character) {
            case 'n':
                result += '\n';
                break;
            case 'r':
                result += '\r';
                break;
            case 't':
                result += '\t';
                break;
            default:
                result += character;
                break;
            }

            escaped = false;
            continue;
        }

        if (character == '\\') {
            escaped = true;
            continue;
        }

        if (character == '"') {
            break;
        }

        result += character;
    }

    return result;
}

}

std::string DictionaryClient::lookup(
    const std::string& word) const {

#ifdef _WIN32
    const std::string encoded =
        url_encode_word(word);

    if (encoded.empty()) {
        return {};
    }

    const std::string command =
        "curl.exe -L -s --fail "
        ""https://api.dictionaryapi.dev/api/v2/entries/en/" +
        encoded +
        """;

    FILE* pipe =
        _popen(command.c_str(), "r");

    if (pipe == nullptr) {
        return {};
    }

    std::string json;
    char buffer[4096];

    while (std::fgets(
        buffer,
        sizeof(buffer),
        pipe) != nullptr) {
        json += buffer;
    }

    const int exit_code =
        _pclose(pipe);

    if (exit_code != 0 || json.empty()) {
        return {};
    }

    const std::string definition =
        extract_json_string(
            json,
            "definition");

    if (definition.empty()) {
        return {};
    }

    const std::string example =
        extract_json_string(
            json,
            "example");

    if (example.empty()) {
        return definition;
    }

    return definition +
        " Example: " +
        example;
#else
    (void)word;
    return {};
#endif
}
