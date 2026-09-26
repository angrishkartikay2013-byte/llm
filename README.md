# ULTRON LLM

ULTRON is a CPU-oriented **Large Language Model engine built from scratch in C++20**. The current model is a small causal Transformer intended for local experiments on modest hardware.

## What ULTRON is

- **AI** is the broad category.
- **LLM** is the neural language model being built here.
- **Chatbot** is the application layer that lets you talk to the model.

ULTRON is therefore a small locally trained **LLM used as a conversational AI chatbot**.

The model does not call ChatGPT or Qwen during normal inference. Ollama is an optional local teacher/data generator; ULTRON remains the student model.

## Current model

The core model currently uses:

- 32-dimensional token embeddings
- 4 attention heads
- 2 causal Transformer blocks
- 128-dimensional feed-forward layers
- 128-token maximum context
- learned token embeddings and output projection
- sinusoidal positional information
- full backpropagation through attention, LayerNorm, GELU, residual paths, both Transformer blocks, and embeddings
- Adam optimization with gradient clipping

This is intentionally small enough to run CPU-only, but it is also far smaller than production LLMs. Training quality is therefore strongly dependent on tokenizer quality, corpus quality, optimization, and model capacity.

## Tokenizer

Fresh models now use a **byte-level BPE tokenizer**.

The tokenizer starts with all 256 possible byte values plus <unk>, then learns deterministic frequent byte-pair merges from the first training corpus.

Important properties:

- Spaces, punctuation, and newlines are represented naturally.
- Unseen words do not collapse to one <unk> token; they fall back to byte/subword pieces.
- BPE merges are frozen after the first tokenizer build so later lessons cannot silently renumber token IDs.
- Newlines are not merged across paragraph boundaries.
- Learned tokens are capped at 16 bytes to reduce tiny-model memorization of long corpus-specific phrases.
- Tokenizer checkpoints include the learned merge table.
- Legacy word-tokenizer checkpoints remain readable for compatibility.

Because token IDs and model weights depend on the tokenizer, **build a fresh checkpoint after the BPE upgrade** rather than judging the new tokenizer with an old word-tokenizer checkpoint.

## Build

From the repository root:

    cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

The smoke tests now cover tensor math, BPE round-tripping, tokenizer serialization, Adam serialization, Transformer backward finiteness, learned-answer memory, and exact training-resume behavior.

## Train from scratch

The safest first experiment after a tokenizer/model upgrade is:

    .\scripts\train.ps1 -Fresh -Epochs 20

The script uses models/ultron_bpe.bin by default so an older models/ultron.bin checkpoint cannot accidentally become the starting point.

It also evaluates the saved checkpoint automatically after training and reports:

- mean loss
- perplexity
- accuracy

That makes it much easier to detect a run that compiled successfully but did not actually improve the model.

To train directly:

    .\build\ultron.exe --train data\train.txt --epochs 20 --lr 0.001 --save models\ultron_bpe.bin

## Continue training

Direct CLI continuation is explicit:

    .\build\ultron.exe --load models\ultron_bpe.bin --train data\train.txt --epochs 20 --lr 0.001 --save models\ultron_bpe.bin

The PowerShell training script now resumes automatically when its checkpoint already exists:

    .\scripts\train.ps1 -Epochs 20

Use:

    .\scripts\train.ps1 -Fresh -Epochs 20

when you intentionally want a new model.

New checkpoints store the tokenizer, model weights, learned question-answer associations, and Adam optimizer state. A dedicated smoke test verifies that:

    2 uninterrupted epochs

produces the same model as:

    1 epoch -> save checkpoint -> load -> 1 more epoch

This specifically guards against the common mistake of thinking a run is continuing when the optimizer or model state actually restarted.

## Generate

For a clean inference test that does not train on the model's own answers:

    .\build\ultron.exe --load models\ultron_bpe.bin --max-tokens 30 --temperature 0.7 --top-k 5 --no-online-learning

For a deterministic regression-style sample:

    .\build\ultron.exe --load models\ultron_bpe.bin --max-tokens 30 --temperature 0.2 --top-k 1 --no-online-learning

## Conversation learning

Interactive mode keeps recent conversation context in:

    data/conversations.txt

Explicit teaching is supported:

    teach what is attention => attention combines information from relevant positions in context.

The explicit teach path writes a learned question-answer association into the checkpoint so that exact taught questions can be recalled reliably.

Automatic online learning is intentionally separate from clean inference. For experiments and evaluation, use:

    --no-online-learning

This avoids immediately training on ULTRON's own possibly incorrect generated responses.

## Evaluation

To evaluate a saved model on a corpus:

    .\build\ultron.exe --load models\ultron_bpe.bin --eval data\train.txt --no-online-learning

The reported metrics are mean loss, perplexity, and next-token accuracy.

## External English data

Download a CC0 English sentence corpus from Tatoeba:

    .\scripts\download_tatoeba.ps1 -MaxSentences 100000

Then:

    .\scripts\train.ps1 -ExternalCorpus -Epochs 1

The training script resumes an existing checkpoint unless -Fresh is supplied.

## Ollama teacher

Ollama can act as a temporary local teacher while ULTRON remains the student.

One-shot bootstrap:

    .\scripts\ollama_bootstrap.ps1 -Model "qwen3:8b" -Epochs 1

The bootstrap now resumes models/ultron_bpe.bin by default. Use -Fresh for an intentional reset.

Cumulative replay teacher:

    .\scripts\ollama_teacher.ps1 -Model "qwen3:8b" -Rounds 10 -ReplayEpochs 5

Each lesson is added to the replay corpus, and ULTRON is retrained on the base corpus plus all accumulated lessons. Ollama is only the temporary lesson generator; ULTRON remains the local student checkpoint.

## Design goal

The project is intentionally being built in stages:

1. reliable tokenizer and text representation
2. reliable forward/backward training
3. checkpointable model and optimizer state
4. better language data
5. teacher-generated lessons and replay
6. larger model capacity as hardware permits
7. richer ULTRON assistant features around the LLM

The project should be judged by reproducible training curves, evaluation metrics, and generation tests—not by the number of epochs alone.
