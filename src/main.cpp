#include "model.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
struct Options {
    std::string train_file;
    std::string eval_file;
    std::string load_file;
    std::string save_file;
    std::size_t epochs = 1;
    float learning_rate = 0.003f;
    std::size_t max_new_tokens = 16;
    float temperature = 0.8f;
    std::size_t top_k = 8;
    unsigned int seed = 42;
};

std::string value_after(
    int& index,
    int argc,
    char** argv,
    const char* name) {

    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string("Missing value for ") + name);
    }

    return argv[++index];
}

Options parse(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        if (argument == "--train") {
            options.train_file =
                value_after(i, argc, argv, "--train");
        } else if (argument == "--eval") {
            options.eval_file =
                value_after(i, argc, argv, "--eval");
        } else if (argument == "--load") {
            options.load_file =
                value_after(i, argc, argv, "--load");
        } else if (argument == "--save") {
            options.save_file =
                value_after(i, argc, argv, "--save");
        } else if (argument == "--epochs") {
            options.epochs = std::stoull(
                value_after(i, argc, argv, "--epochs"));
        } else if (argument == "--lr") {
            options.learning_rate = std::stof(
                value_after(i, argc, argv, "--lr"));
        } else if (argument == "--max-tokens") {
            options.max_new_tokens = std::stoull(
                value_after(i, argc, argv, "--max-tokens"));
        } else if (argument == "--temperature") {
            options.temperature = std::stof(
                value_after(i, argc, argv, "--temperature"));
        } else if (argument == "--top-k") {
            options.top_k = std::stoull(
                value_after(i, argc, argv, "--top-k"));
        } else if (argument == "--seed") {
            options.seed = static_cast<unsigned int>(
                std::stoul(
                    value_after(i, argc, argv, "--seed")));
        } else if (
            argument == "--help" ||
            argument == "-h") {

            std::cout
                << "ULTRON LLM\n\n"
                << "--train FILE --eval FILE\n"
                << "--epochs N --lr RATE\n"
                << "--load FILE --save FILE\n"
                << "--max-tokens N --temperature T\n"
                << "--top-k K --seed N\n";

            std::exit(0);
        } else {
            throw std::invalid_argument(
                "Unknown argument: " + argument);
        }
    }

    return options;
}

std::string read_file(
    const std::string& path) {

    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "Cannot open: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}
}

int main(int argc, char** argv) {
    try {
        const Options options =
            parse(argc, argv);

        ULTRONModel model;

        if (!options.load_file.empty()) {
            if (!model.load_checkpoint(
                    options.load_file)) {
                std::cerr
                    << "Failed to load checkpoint: "
                    << options.load_file
                    << '\n';
                return 1;
            }

            std::cout
                << "Checkpoint loaded.\n";
        }

        if (!options.train_file.empty()) {
            const std::string text =
                read_file(options.train_file);

            std::cout
                << "Training for "
                << options.epochs
                << " epoch(s)...\n";

            const float loss =
                model.train(
                    text,
                    options.epochs,
                    options.learning_rate);

            std::cout
                << "Training complete. Mean loss: "
                << loss
                << '\n';

            if (!options.save_file.empty()) {
                if (!model.save_checkpoint(
                        options.save_file)) {
                    std::cerr
                        << "Failed to save checkpoint: "
                        << options.save_file
                        << '\n';
                    return 1;
                }

                std::cout
                    << "Checkpoint saved.\n";
            }
        }

        if (!options.eval_file.empty()) {
            const ModelEvaluation metrics =
                model.evaluate(
                    read_file(options.eval_file));

            std::cout
                << "Evaluation samples: "
                << metrics.samples
                << '\n'
                << "Mean loss: "
                << metrics.mean_loss
                << '\n'
                << "Perplexity: "
                << metrics.perplexity
                << '\n'
                << "Accuracy: "
                << metrics.accuracy
                << '\n';
        }

        std::cout
            << "ULTRON ready. Type 'exit' to quit.\n";

        std::string input;

        while (true) {
            std::cout << "> ";

            if (!std::getline(
                    std::cin,
                    input)) {
                break;
            }

            if (input == "exit") {
                break;
            }

            if (input.empty()) {
                continue;
            }

            std::cout
                << "ULTRON: "
                << model.generate(
                    input,
                    options.max_new_tokens,
                    options.temperature,
                    options.top_k,
                    options.seed)
                << '\n';
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "ULTRON error: "
            << error.what()
            << '\n';

        return 1;
    }
}
