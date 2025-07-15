# Recipes

This directory contains YAML configuration files for running fairness-aware language model evaluation experiments.

## Available Recipes

- **Base Configuration**: [`base.yaml`](base.yaml) - Basic template configuration
- **Fairness Example**: [`fairness_example.yaml`](fairness_example.yaml) - Complete example using bias detection PRM on BBQ dataset

## Usage

Run experiments by specifying the recipe file:

```bash
# Run the fairness example
python scripts/test_time_compute.py recipes/fairness_example.yaml

# Run with base config and overrides
python scripts/test_time_compute.py recipes/base.yaml --model_path=meta-llama/Llama-3.2-3B-Instruct --dataset.name=heegyu/bbq
```

## Configuration Structure

Each recipe contains:
- **model**: Model configuration including PRM paths for bias detection
- **dataset**: Dataset selection (BBQ, COMPAS, Bias in Bios, etc.)
- **search**: Search algorithm settings (Best-of-N, beam search, DVTS)
- **output**: Output configuration including fairness metrics computation


