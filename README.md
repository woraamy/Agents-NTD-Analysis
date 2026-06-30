# NTD Analysis

This folder keeps the non-AI research workflow for not-to-do instruction analysis.

## Layout

- `scripts/baselines/` contains deterministic baseline scripts.
- `scripts/data_prep/` contains random sampling and sentence splitting scripts.
- `scripts/plots/` contains plotting scripts for manual inspection outputs.
- `manual_inspect_dataset/`, `split_dataset/`, `raw_datasets/`, and `collected_datasets/` contain data artifacts.

## Current Scripts

- `python scripts/data_prep/random_sample_and_split_sentences.py`
- `python scripts/baselines/train_regex_ntd_detection_colab.py`
- `python scripts/plots/plot_ntd_distribution.py`
- `python scripts/plots/plot_ntd_boxplot.py`

AI, LLM, neural fine-tuning, and provider API scripts have been removed from this workspace.
