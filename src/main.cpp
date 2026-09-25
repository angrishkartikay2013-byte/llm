#include "model.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
struct Options {
    std::string train_file; std::string load_file; std::string save_file;
    std::size_t epochs = 1; float learning_rate = 0.003f;
    std::size_t max_new_tokens = 16; float temperature = 0.8f;
    std::size_t top_k = 8; unsigned int seed = 42;
};
std::string value_after(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("Missing value for ") + name);
    return argv[++i];
}
std::size_t size_value(const std::string& value) { return std::stoull(value); }
float float_value(const std::string& value) { return std::stof(value); }
unsigned int uint_value(const std::string& value) { return static_cast<unsigned int>(std::stoul(value)); }
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--train") o.train_file = value_after(i, argc, argv, "--train");
        else if (a == "--load") o.load_file = value_after(i, argc, argv, "--load");
        else if (a == "--save") o.save_file = value_after(i, argc, argv, "--save");
        else if (a == "--epochs") o.epochs = size_value(value_after(i, argc, argv, "--epochs"));
        else if (a == "--lr") o.learning_rate = float_value(value_after(i, argc, argv, "--lr"));
        else if (a == "--max-tokens") o.max_new_tokens = size_value(value_after(i, argc, argv, "--max-tokens"));
        else if (a == "--temperature") o.temperature = float_value(value_after(i, argc, argv, "--temperature"));
        else if (a == "--top-k") o.top_k = size_value(value_after(i, argc, argv, "--top-k"));
        else if (a == "--seed") o.seed = uint_value(value_after(i, argc, argv, "--seed"));
        else if (a == "--help" || a == "-h") {
            std::cout << "ULTRON LLM\n\n--load FILE --train FILE --epochs N --lr RATE --save FILE\n--max-tokens N --temperature T --top-k K --seed N\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown argument: " + a);
    }
    return o;
}
std::string read_file(const std::string& path) {
    std::ifstream input(path); if (!input) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream buffer; buffer << input.rdbuf(); return buffer.str();
}
}

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        ULTRONModel model;
        if (!options.load_file.empty()) {
            if (!model.load_checkpoint(options.load_file)) { std::cerr << "Failed to load checkpoint: " << options.load_file << '\n'; return 1; }
            std::cout << "Checkpoint loaded.\n";
        }
        if (!options.train_file.empty()) {
            const std::string text = read_file(options.train_file);
            std::cout << "Training for " << options.epochs << " epoch(s)...\n";
            model.train(text, options.epochs, options.learning_rate);
            std::cout << "Training complete.\n";
            if (!options.save_file.empty()) {
                if (!model.save_checkpoint(options.save_file)) { std::cerr << "Failed to save checkpoint: " << options.save_file << '\n'; return 1; }
                std::cout << "Checkpoint saved.\n";
            }
        }
        std::cout << "ULTRON ready. Type 'exit' to quit.\n";
        std::string input;
        while (true) {
            std::cout << "> "; if (!std::getline(std::cin, input)) break;
            if (input == "exit") break; if (input.empty()) continue;
            std::cout << "ULTRON: " << model.generate(input, options.max_new_tokens, options.temperature, options.top_k, options.seed) << '\n';
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "ULTRON error: " << e.what() << '\n'; return 1; }
}
