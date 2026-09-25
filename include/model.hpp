#pragma once

#include <memory>
#include <string>

class ULTRONModel {
public:
    ULTRONModel();
    ~ULTRONModel();

    ULTRONModel(const ULTRONModel&) = delete;
    ULTRONModel& operator=(const ULTRONModel&) = delete;
    ULTRONModel(ULTRONModel&&) noexcept;
    ULTRONModel& operator=(ULTRONModel&&) noexcept;

    void train(const std::string& text);

    std::string generate(const std::string& prompt) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};