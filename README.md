# ULTRON LLM

ULTRON is a CPU-oriented Large Language Model engine built from scratch in C++20.

## Build

Use:

    cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

## Train

Train on the bundled local corpus:

    .\build\ultron.exe --train data/train.txt --epochs 5 --lr 0.003 --save models\ultron.bin

Run a saved model:

    .\build\ultron.exe --load models\ultron.bin --max-tokens 30

## Conversation learning

Interactive mode keeps a small rolling conversation history in:

    data/conversations.txt

Each completed exchange receives a small online learning update and the live checkpoint is saved to:

    models/ultron_live.bin

To explicitly teach ULTRON a fact or response:

    teach what is attention => attention combines information from relevant positions in context.

To ask the connected dictionary:

    define attention

The dictionary command queries the public Free Dictionary API at runtime; definitions are not hardcoded into the model.

## External training data

Download an English CC0 sentence corpus from Tatoeba:

    .\scripts\download_tatoeba.ps1 -MaxSentences 100000

Then train with:

    .\scripts\train.ps1 -ExternalCorpus -Epochs 1

Downloaded corpora and local model files are ignored by Git.

Google Research also published the One Billion Word Benchmark, an approximately one-billion-word language-modeling corpus. It is useful for research, but it is far too large for the current CPU-only ULTRON experiments, so ULTRON does not automatically download it.


## Ollama one-shot distillation

ULTRON can temporarily use an installed local Ollama model to generate additional training sentences, train the local C++ model on them, and then unload the Ollama model.

Run:

    .\scripts\ollama_bootstrap.ps1 -Model "qwen3:8b" -Epochs 1

The script talks only to Ollama on `http://localhost:11434`, writes the generated corpus to `data/external/ollama_distill.txt`, trains ULTRON, saves `models/ultron.bin`, and then unloads the model and stops the daemon if the script started it. The default bootstrap is deliberately small: one batch of 30 short sentences, suitable for a quick CPU experiment.

This is knowledge distillation/data synthesis: Ollama is the temporary teacher, while ULTRON remains the standalone model after the bootstrap finishes.
