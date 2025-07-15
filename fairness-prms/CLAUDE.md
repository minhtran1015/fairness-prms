# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fairness-aware language model evaluation framework that implements test-time compute scaling with bias detection. The project focuses on improving fairness in language model outputs through Best-of-N sampling, beam search, and diverse verifier tree search (DVTS) combined with specialized fairness reward models.

## Key Components

- **Test-time Search Algorithms**: Located in `src/sal/search/` - implements Best-of-N, beam search, and DVTS
- **Fairness Reward Models (PRMs)**: Located in `src/sal/models/prm/` - specialized models for bias detection
- **Configuration System**: Located in `src/sal/config/` - YAML-based configuration with dataclass definitions
- **Dataset Utilities**: Located in `src/sal/utils/` - handles fairness benchmark datasets (BBQ, COMPAS, Bias in Bios, etc.)

## Development Commands

### Installation
```bash
conda create -n sal python=3.10 && conda activate sal
pip install -e '.[dev]'
```

### Code Quality
```bash
# Format code and fix imports
make style

# Check code quality
make quality

# Manual ruff commands
ruff check --select I --fix src scripts tests setup.py
ruff format src scripts tests setup.py
```

### Running Experiments
```bash
# Run with recipe (recommended)
python scripts/test_time_compute.py --recipe "llama_best_of_n"

# Run with custom config
python scripts/test_time_compute.py --config path/to/config.yaml

# Run with recipe override
python scripts/test_time_compute.py recipes/base.yaml --model_path meta-llama/Llama-3.2-3B-Instruct
```

## Architecture

### Configuration System
- `Config` dataclass contains `ModelConfig`, `DatasetConfig`, `SearchConfig`, and `OutputConfig`
- YAML recipes in `recipes/` directory provide presets for different experiments
- Config loader supports both legacy path-based and modern YAML-based configurations

### Model Registry
- PRM models are registered in `src/sal/models/registry.py`
- `BiasDetectionPRM` is the main fairness-aware reward model
- Model loading handles tokenizer setup and device placement

### Search Algorithms
- `best_of_n`: Generates N candidates and selects highest-scoring based on PRM
- `beam_search`: Maintains top-K candidates during generation
- `dvts`: Diverse verifier tree search with multiple verification steps

### Dataset Support
- BBQ (Bias Benchmark Questions) - primary fairness benchmark
- COMPAS - recidivism risk assessment
- Bias in Bios - occupation classification
- Civil Comments - content moderation
- Adult Census - income prediction

## Key Files

- `scripts/test_time_compute.py`: Main entry point for running experiments
- `src/sal/config/base.py`: Configuration dataclass definitions
- `src/sal/models/registry.py`: PRM model registry
- `src/sal/search/best_of_n.py`: Best-of-N search implementation
- `src/sal/utils/data.py`: Dataset loading and preprocessing
- `recipes/base.yaml`: Base configuration template

## Development Notes

- Uses vLLM for efficient inference
- Supports multiple PRM models with weighted combination
- Fairness metrics computed automatically (demographic parity, equalized odds, etc.)
- Results can be pushed to Hugging Face Hub for sharing
- GPU memory utilization configurable via `gpu_memory_utilization` parameter