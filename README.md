# NTD Analysis

This folder keeps the research workflow for not-to-do instruction analysis.

## Layout

- `scripts/data_prep/` contains random sampling and sentence splitting scripts.
- `scripts/stats/` contains statistics scripts for inspected data.
- `scripts/plots/` contains plotting scripts for manual inspection outputs.
- `manual_inspect_dataset/`, `split_dataset/`, `raw_datasets/`, and `collected_datasets/` contain data artifacts.

## Current Scripts

- `./env/bin/python scripts/data_prep/random_sample_and_split_sentences.py`
- `./env/bin/python scripts/stats/compute_ntd_statistics.py`
- `./env/bin/python scripts/plots/plot_ntd_distribution.py`
- `./env/bin/python scripts/plots/plot_ntd_boxplot.py`