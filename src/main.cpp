#include "conversation.hpp"
#include "dictionary.hpp"
#include "model.hpp"
#include "self_test.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>

namespace {
struct Options {
    std::string train_file;
    std::string eval_file;
    std::string load_file;
    std::string save_file;
    std::size_t epochs = 1;
    float learning_rate = 0.001f;
    std::size_t max_new_tokens = 16;
    float temperature = 0.8f;
    std::size_t top_k = 8;
    unsigned int seed = 42;
    std::size_t save_every = 0;
    bool self_test = false;
    bool online_learning = false;
    bool interactive = true;
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
        } else if (argument == "--save-every") {
            options.save_every = std::stoull(
                value_after(i, argc, argv, "--save-every"));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else if (argument == "--online-learning") {
            options.online_learning = true;
        } else if (argument == "--no-online-learning") {
            options.online_learning = false;
        } else if (argument == "--non-interactive") {
            options.interactive = false;
        } else if (
            argument == "--help" ||
            argument == "-h") {

            std::cout
                << "ULTRON LLM\n\n"
                << "--train FILE --eval FILE\n"
                << "--epochs N --lr RATE\n"
                << "--load FILE --save FILE\n"
                << "--max-tokens N --temperature T\n"
                << "--top-k K --seed N\n"
                << "--save-every N\n"
                << "--self-test\n"
                << "--online-learning --no-online-learning\n"
                << "--non-interactive\n";

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

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };

    value.erase(
        value.begin(),
        std::find_if(
            value.begin(),
            value.end(),
            not_space));

    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            not_space).base(),
        value.end());

    return value;
}

bool starts_with(
    const std::string& value,
    const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}
}

int main(int argc, char** argv) {
    try {
        const Options options =
            parse(argc, argv);

        if (options.self_test) {
            return run_ultron_smoke_tests();
        }

        ULTRONModel model;
        ConversationStore conversations;
        DictionaryClient dictionary;

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

            const auto checkpoint_progress =
                [&](std::size_t epoch, float) {
                    if (options.save_file.empty() ||
                        options.save_every == 0 ||
                        epoch % options.save_every != 0) {
                        return;
                    }

                    if (!model.save_checkpoint(
                            options.save_file)) {
                        throw std::runtime_error(
                            "Failed to save checkpoint during training: " +
                            options.save_file);
                    }

                    std::cout
                        << "Checkpoint saved at epoch "
                        << epoch
                        << ".\n";
                };

            const float loss =
                model.train(
                    text,
                    options.epochs,
                    options.learning_rate,
                    checkpoint_progress);

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

        if (!options.interactive) {
            return 0;
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

            if (starts_with(input, "teach ")) {
                const std::string lesson =
                    trim_copy(input.substr(6));
                const std::size_t separator =
                    lesson.find("=>");

                if (separator == std::string::npos) {
                    std::cout
                        << "ULTRON: Use teach question => answer\n";
                    continue;
                }

                const std::string question =
                    trim_copy(lesson.substr(0, separator));
                const std::string answer =
                    trim_copy(lesson.substr(separator + 2));

                if (question.empty() || answer.empty()) {
                    std::cout
                        << "ULTRON: Both sides of => are required.\n";
                    continue;
                }

                const float loss =
                    model.train(
                        "USER: " + question +
                        "\nULTRON: " + answer,
                        2,
                        options.learning_rate * 0.2f);

                conversations.append(
                    question,
                    answer);

                model.save_checkpoint(
                    "models/ultron_live.bin");

                std::cout
                    << "ULTRON learned. loss="
                    << loss
                    << '\n';
                continue;
            }

            if (starts_with(input, "define ")) {
                const std::string word =
                    trim_copy(input.substr(7));

                const std::string definition =
                    dictionary.lookup(word);

                if (definition.empty()) {
                    std::cout
                        << "ULTRON: I couldn't retrieve a definition for '"
                        << word
                        << "'.\n";
                } else {
                    std::cout
                        << "ULTRON: "
                        << definition
                        << '\n';
                }

                continue;
            }

            std::string prompt = input;
            const std::string memory =
                conversations.recent_context(4);

            if (!memory.empty()) {
                prompt =
                    memory +
                    "USER: " +
                    input +
                    "\nULTRON:";
            } else {
                prompt =
                    "USER: " +
                    input +
                    "\nULTRON:";
            }

            const std::string full_generation =
                model.generate(
                    prompt,
                    options.max_new_tokens,
                    options.temperature,
                    options.top_k,
                    options.seed);

            std::string response =
                full_generation.rfind(prompt, 0) == 0
                    ? full_generation.substr(prompt.size())
                    : full_generation;

            response = trim_copy(response);

            std::cout
                << "ULTRON: "
                << response
                << '\n';

            // Conversation history is always persisted, but ULTRON only
            // trains on its own generated responses when explicitly enabled.
            conversations.append(
                input,
                response);

            if (options.online_learning &&
                !response.empty()) {

                model.train(
                    "USER: " + input +
                    "\nULTRON: " + response,
                    1,
                    options.learning_rate * 0.1f);

                model.save_checkpoint(
                    "models/ultron_live.bin");
            }
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
