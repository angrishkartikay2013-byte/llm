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
- 256-token maximum context
- learned token embeddings and output projection
- sinusoidal positional information
- full backpropagation through attention, LayerNorm, GELU, residual paths, both Transformer blocks, and embeddings
- Adam optimization with gradient clipping

This is intentionally small enough to run CPU-only, but it is also far smaller than production LLMs. Training quality is therefore strongly dependent on tokenizer quality, corpus quality, optimization, and model capacity.

## Tokenizer

Fresh models now use a **byte-level BPE tokenizer**.

The tokenizer starts with all 256 possible byte values plus <unk>, then learns deterministic frequent byte-pair merges from the first training corpus. Fresh BPE models use boundary-aware tokenization: merges happen inside words and other local units, while spaces, tabs, punctuation, and line breaks remain explicit boundaries.

Important properties:

- Spaces, tabs, punctuation, and newlines remain explicit boundaries so generated text can learn readable layout.
- Unseen words do not collapse to one <unk> token; they fall back to byte/subword pieces.
- BPE merges are frozen after the first tokenizer build so later lessons cannot silently renumber token IDs.
- BPE merges never cross word, whitespace, punctuation, or newline boundaries.
- Learned tokens are capped at 16 bytes to reduce tiny-model memorization of long corpus-specific phrases.
- Tokenizer checkpoints include the learned merge table.
- Legacy word-tokenizer checkpoints remain readable for compatibility.

Because token IDs and model weights depend on the tokenizer, **build a fresh checkpoint after the BPE upgrade** rather than judging the new tokenizer with an old word-tokenizer checkpoint.

## Build

From the repository root:

    cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

The smoke tests now cover tensor math, boundary-aware BPE round-tripping, tokenizer serialization, Adam serialization, Transformer backward finiteness, learned-answer memory, and exact training-resume behavior. The training script also refuses obviously undersized corpora and checks for the conversation and punctuation patterns expected in the main corpus.

## Train from scratch

The repository now ships a language-focused training corpus plus a held-out validation corpus. The training corpus contains explicit lessons for spacing, punctuation, dialogue turns, paragraph structure, reasoning language, commands, formal/casual tone, numbers, and natural narration.

The safest first experiment after a tokenizer/model upgrade is:

    .\scripts\train.ps1 -Fresh -Epochs 20 -SaveEvery 10

The script uses models/ultron_bpe.bin by default so an older models/ultron.bin checkpoint cannot accidentally become the starting point.

It also evaluates the saved checkpoint automatically after training on both the training corpus and the held-out validation corpus and reports:

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

    .\scripts\train.ps1 -Epochs 20 -SaveEvery 10

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

Automatic online learning is disabled by default and intentionally separate from clean inference. To enable it explicitly, use:

    --online-learning

For clean experiments, use --no-online-learning explicitly as well. This avoids immediately training on ULTRON's own possibly incorrect generated responses.

## Evaluation

To evaluate a saved model on a corpus:

    .\build\ultron.exe --load models\ultron_bpe.bin --eval data\train.txt --no-online-learning

The reported metrics are mean loss, perplexity, and next-token accuracy.

## External English data

Download a CC0 English sentence corpus from Tatoeba:

    .\scripts\download_tatoeba.ps1 -MaxSentences 100000

Then:

    .\scripts\train.ps1 -ExternalCorpus -Epochs 1

The training script resumes an existing checkpoint unless -Fresh is supplied. It saves a recovery checkpoint every 10 epochs by default.

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

## Autonomous internet learning

ULTRON can run a controlled autonomous learning loop on Windows without a pretrained-model wrapper.

The controller downloads an English CC0 sentence corpus from Tatoeba over the internet, mixes a random replay sample from the seed corpus with a fresh internet batch, trains for one epoch, and periodically runs a basic-English generation gate. The replay mix is intentional: training only on sequential new batches could make this small model forget earlier language.

Run it with:

    .\scripts\autolearn.ps1 -SeedData data\train.txt -MaxRounds 100 -Speed 1

For a fresh autonomous checkpoint:

    .\scripts\autolearn.ps1 -SeedData data\train.txt -Fresh -MaxRounds 100 -Speed 1

Use your own corpus as the seed by passing its path with `-SeedData`, for example:

    .\scripts\autolearn.ps1 -SeedData data\ultron_training_corpus.txt -Fresh

Autonomous controls while a training batch is running:

- `S` saves a safety copy at the next completed checkpoint.
- `T` queues an immediate basic-English test after the current batch.
- `P` pauses or resumes after the current batch.
- `X` stops after the current batch and saves a safe checkpoint.

The autonomous learner stops automatically when its basic-English gate reaches the configured threshold for the configured number of consecutive tests. The default is 75/100 for 3 consecutive passing tests. The gate checks eight prompts for readable output, reasonable length, alphabetic content, control-token leakage, repeated characters, repeated 3-grams, and topic-keyword hits on factual prompts.

This gate is deliberately conservative about its wording: it is a measurable baseline for basic English generation, not proof that ULTRON is intelligent or generally capable. The program never uses ULTRON's own generated answers as training data in this mode.

The default internet source is Tatoeba's CC0 English export. Downloaded corpora and round files remain under `data/external/`, which is ignored by Git. Model checkpoints remain under `models/`, which is also ignored by Git.


## Hugging Face control experiment

ULTRON can build a bounded control corpus from the Hugging Face HuggingFaceH4/ultrachat_200k dataset without downloading the full multi-gigabyte dataset.

The downloader uses the Hugging Face Dataset Viewer /rows API and extracts complete conversations from the train_sft and test_sft splits. The generated files stay under data/external/ and are ignored by Git.

Download a modest control set first:

    .\scripts\download_ultrachat.ps1 -TrainConversations 3000 -ValidationConversations 500

Then run a fresh experiment:

    .\scripts\train_ultrachat_control.ps1 -Fresh -Epochs 5 -Speed 1

For a faster smoke experiment:

    .\scripts\download_ultrachat.ps1 -TrainConversations 1000 -ValidationConversations 200

    .\scripts\train_ultrachat_control.ps1 -Fresh -Epochs 1 -Speed 5

### Why this is a control experiment

The model architecture, C++ trainer, tokenizer implementation, optimizer, checkpoint format, and generation code remain unchanged. The controlled change is the corpus.

The downloader converts complete exchanges into USER and ULTRON turns and keeps a separate held-out test corpus.

UltraChat 200k is a filtered English conversational dataset with separate supervised fine-tuning train/test splits. The Hugging Face dataset card says the released version was filtered, truecased, and used for Zephyr-7B-beta training. The original UltraChat dialogues were generated by ChatGPT, so this is a standardized control dataset, not a human-written corpus.

This experiment is designed to answer a specific engineering question: does ULTRON's current training code learn a substantially cleaner external corpus than the previous corpus?


## CPU parallel training

ULTRON automatically detects the available hardware threads and uses them for independent training windows. On an 8-logical-processor CPU, training batches use up to 8 workers while shared model parameters remain synchronized between batches.

The BPE tokenizer also parallelizes pair-frequency counting and merge application across the detected hardware threads. Nested Transformer threading is disabled inside outer training workers to avoid spawning excessive numbers of threads.

Training output reports the detected CPU worker count and the batch worker count.