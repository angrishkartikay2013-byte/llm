# ULTRON LLM

ULTRON is a CPU-oriented Large Language Model engine built from scratch in C++20. The current model uses a two-block causal Transformer and trains embeddings, both Transformer blocks, and the output projection end-to-end. The tokenizer is a learned byte-level BPE tokenizer with ASCII case normalization, so unseen words do not automatically collapse to a single <unk> token.

## Build

Use:

    cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

## Train

Train on the bundled local corpus. Training sweeps across the entire corpus in overlapping context windows, so long datasets are not truncated to only its final context:

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

The script talks only to Ollama on http://localhost:11434, writes the generated corpus to data/external/ollama_distill.txt, trains ULTRON, saves models/ultron.bin, and then unloads the model and stops the daemon if the script started it. The default bootstrap is deliberately small: one batch of 30 short sentences, suitable for a quick CPU experiment.

This is knowledge distillation/data synthesis: Ollama is the temporary teacher, while ULTRON remains the standalone model after the bootstrap finishes.

## Training architecture

The training path performs full gradient backpropagation through the output projection, both Transformer blocks, causal multi-head attention, GELU, LayerNorm, residual paths, and token embeddings. Gradients are clipped before Adam updates. Embedding gradients use the same per-window normalization as the upstream hidden-state gradients. The smoke test also exercises the Transformer backward pass directly.

Generation supports a clean inference mode that prevents generated responses from being fed back into online learning:

    .\\build\\ultron.exe --load models\\ultron.bin --max-tokens 20 --no-online-learning

Model checkpoints written by the current training path use checkpoint version 4 and include the learned tokenizer merges plus both Transformer blocks. Re-train a new checkpoint after pulling tokenizer or architecture changes rather than judging a newly built binary with an older checkpoint.

## Ollama teacher with replay

Run:

    .\scripts\ollama_teacher.ps1 -Model "qwen3:8b" -Rounds 5

The teacher now uses cumulative replay training. Each round asks Ollama for a new lesson, shows ULTRON's answer before learning, permanently appends the question/answer to data/external/ollama_teacher.txt, rebuilds a combined corpus from data/train.txt plus every accumulated lesson, and retrains ULTRON on that full replay corpus. This prevents the student update from focusing only on the newest lesson.

By default each lesson receives 5 replay epochs at learning rate 0.001. Adjust them with:

    .\scripts\ollama_teacher.ps1 -Model "qwen3:8b" -Rounds 10 -ReplayEpochs 5 -LearningRate 0.001

The teacher also asks Ollama to avoid recently used questions. Use -ResetTeacherCorpus to start a fresh teacher run and discard the previous local lesson corpus/live checkpoint:

    .\scripts\ollama_teacher.ps1 -Model "qwen3:8b" -Rounds 100 -ReplayEpochs 5 -ResetTeacherCorpus

The accumulated student checkpoint is saved as models/ultron_live.bin. The replay corpus and combined training corpus are runtime files and are ignored by Git.

No factual answers are hard-coded. Ollama supplies the lessons dynamically through its local API; ULTRON remains the student model.
