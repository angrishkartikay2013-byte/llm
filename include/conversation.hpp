#pragma once

#include <cstddef>
#include <string>

class ConversationStore {
public:
    explicit ConversationStore(
        std::string path = "data/conversations.txt");

    bool append(
        const std::string& user,
        const std::string& assistant) const;

    std::string recent_context(
        std::size_t max_turns = 4) const;

private:
    std::string path_;
};
