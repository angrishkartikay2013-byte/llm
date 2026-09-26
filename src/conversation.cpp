#include "conversation.hpp"

#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

ConversationStore::ConversationStore(
    std::string path)
    : path_(std::move(path)) {}

bool ConversationStore::append(
    const std::string& user,
    const std::string& assistant) const {

    std::ofstream output(
        path_,
        std::ios::app);

    if (!output) {
        return false;
    }

    output
        << "USER: "
        << user
        << '\n'
        << "ULTRON: "
        << assistant
        << '\n';

    return static_cast<bool>(output);
}

std::string ConversationStore::recent_context(
    std::size_t max_turns) const {

    if (max_turns == 0) {
        return {};
    }

    std::ifstream input(path_);

    if (!input) {
        return {};
    }

    std::deque<std::pair<std::string, std::string>> turns;
    std::string line;
    std::string user;
    std::string assistant;

    while (std::getline(input, line)) {
        if (line.rfind("USER: ", 0) == 0) {
            user = line.substr(6);
        } else if (line.rfind("ULTRON: ", 0) == 0) {
            assistant = line.substr(8);

            if (!user.empty()) {
                turns.emplace_back(user, assistant);

                while (turns.size() > max_turns) {
                    turns.pop_front();
                }
            }

            user.clear();
            assistant.clear();
        }
    }

    std::ostringstream context;

    for (const auto& turn : turns) {
        context
            << "USER: "
            << turn.first
            << '\n'
            << "ULTRON: "
            << turn.second
            << '\n';
    }

    return context.str();
}
