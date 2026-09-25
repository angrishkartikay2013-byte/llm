# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

The repository now includes dataset utilities and standard language-model evaluation metrics: mean cross-entropy loss, perplexity, and next-token accuracy.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Train

```text
build/ultron --train data/train.txt --epochs 5 --lr 0.003 --save models/ultron.bin
```

## Evaluate

The metrics library can evaluate batches of model logits against target token IDs.

