#pragma once

#include <string>

class DictionaryClient {
public:
    std::string lookup(
        const std::string& word) const;
};
