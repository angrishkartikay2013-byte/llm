# External training data

This directory is for corpora downloaded at runtime. Downloaded files are ignored by Git.

## Tatoeba CC0 English

Use:

    .\scripts\download_tatoeba.ps1 -MaxSentences 100000

The script retrieves the current English CC0 sentence export from Tatoeba, extracts the sentence text, and writes a plain-text corpus for ULTRON.

Tatoeba provides both CC BY 2.0 FR material and a separate CC0 sentence subset. Prefer the CC0 subset when you want a simple redistribution-friendly training source.

## Google language-model data

Google Research published the One Billion Word Benchmark, a corpus with almost one billion words for language-model research. It is useful as a research reference, but it is far too large for the current CPU-focused ULTRON experiments.

Large external corpora should stay out of the Git repository and be downloaded only when their licenses and redistribution terms are appropriate.