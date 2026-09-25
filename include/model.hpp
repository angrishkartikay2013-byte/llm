#pragma once

#include <string>

class ULTRONModel {
public:
    ULTRONModel();

    void train(const std::string& text);
    std::string generate(const std::string& prompt) const;
};