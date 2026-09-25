# ULTRON LLM

A Large Language Model engine built from scratch in modern C++20.

Windows quick start:

1. Run scripts/build.ps1
2. Run scripts/train.ps1
3. Run scripts/run.ps1

Custom training data can be supplied with the -Data parameter.

Current training updates the output projection and input embeddings while the Transformer remains fixed for the next backpropagation milestone.