# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

## Engine

- Custom tokenizer
- Dense tensor and matrix operations
- Learned embeddings
- Causal multi-head self-attention
- Residual connections
- Layer normalization
- GELU feed-forward network
- Next-token training with Adam
- Temperature and top-k sampling
- Binary checkpoints
- Dataset and model configuration utilities

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Train

```text
cmake --build build
build/ultron --train data/train.txt --epochs 5 --lr 0.003 --save models/ultron.bin
```

## Load

```text
build/ultron --load models/ultron.bin
```
