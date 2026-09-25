# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

The tokenizer now keeps punctuation as separate tokens, making training text boundaries more useful than plain whitespace splitting.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Train

```text
build/ultron --train data/train.txt --epochs 5 --lr 0.003 --save models/ultron.bin
```
