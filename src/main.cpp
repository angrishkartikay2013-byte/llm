#include "model.hpp"

#include <iostream>
#include <string>

int main() {
    ULTRONModel model;

    std::cout << "ULTRON LLM engine" << std::endl;
    std::cout << "Type text to run the local model, or 'exit' to quit.\n";

    std::string input;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, input)) {
            break;
        }

        if (input == "exit") {
            break;
        }

        if (input.empty()) {
            continue;
        }

        std::cout << "ULTRON: " << model.generate(input) << std::endl;
    }

    return 0;
}
