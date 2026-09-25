# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

## Engine

- Custom tokenizer
- Tensor and matrix operations
- Learned embeddings
- Causal multi-head self-attention
- Residual connections
- Layer normalization
- GELU feed-forward network
- Next-token training with Adam
- Temperature and top-k sampling
- Binary checkpoints
- Command-line training and inference

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Train

```bash
build/ultron --train data/train.txt --epochs 5 --lr 0.003 --save models/ultron.bin
```

## Load

```bash
build/ultron --load models/ultron.bin
```

## Generation

```text
--max-tokens N
--temperature T
--top-k K
--seed N
```

Current training updates the output projection; the Transformer remains the next target for end-to-end backpropagation.
