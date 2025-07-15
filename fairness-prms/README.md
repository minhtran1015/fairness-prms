# Fairness Reward Models

Implementation of "Fairness Reward Models" - a framework for improving fairness in language model outputs through test-time search and specialized reward models.

## Overview

This repository implements Best-of-N sampling combined with fairness-aware reward models to improve model performance on bias-sensitive tasks. Our approach focuses on reducing harmful biases while maintaining task performance across multiple fairness benchmarks.

## Installation

Set up the environment:

```bash
conda create -n sal python=3.10 && conda activate sal
pip install -e '.[dev]'
```

## Supported Datasets

Our framework supports evaluation on several fairness benchmarks:

- **BBQ (Bias Benchmark Questions)**: Multiple choice bias evaluation across demographics
- **COMPAS**: Recidivism risk assessment with racial fairness considerations  
- **Bias in Bios**: Occupation classification from biographical text
- **Civil Comments**: Content moderation with fairness constraints

## Quick Start

### Running Best-of-N with Fairness Reward Model

```bash
python scripts/test_time_compute.py --recipe "llama_best_of_n"
```

### Custom Configuration

Create a YAML configuration file:

```yaml
model:
  model_path: "meta-llama/Llama-3.2-1B-Instruct"
  prm_paths: ["zarahall/bias-prm-v3"]  # Fairness-aware reward model
  
dataset:
  name: "heegyu/bbq"  # Bias benchmark
  split: "test"
  
search:
  approach: "best_of_n"
  n: 32  # Generate 32 candidates
  temperature: 0.8
  
output:
  push_to_hub: true
  compute_fairness_metrics: true
```

Then run:
```bash
python scripts/test_time_compute.py --config your_config.yaml
```

## Search Strategy

### Best-of-N Sampling
Generates N completions and selects the highest-scoring one according to the fairness reward model.

## Project Structure

```
src/sal/
├── config/          # Configuration system
│   ├── base.py      # Base configuration classes
│   ├── prompts.py   # Dataset-specific prompts
│   └── loader.py    # Configuration loading
├── models/          # Reward model implementations
│   ├── prm/         # Process reward models
│   │   ├── bias_detection.py  # Fairness-aware PRM
│   │   └── math_shepherd.py   # Math reasoning PRM
│   └── registry.py  # Model registry
├── search/          # Search algorithms
│   ├── best_of_n.py
│   ├── beam_search.py
│   └── diverse_verifier_tree_search.py
└── utils/           # Utilities for scoring and data processing
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{hall2025fairnessrewardmodels
  title={Guiding LLM Decision-Making with Fairness Reward Models},
  author={Zara Hall and Thomas Zollo and Melanie Subbiah and Kathleen McKeown and Richard Zemel},
  journal={ArXiv},
  year={2025}
}
```

This repository is based on the HuggingFace implementation of test-time compute scaling:

```bibtex
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

And the original DeepMind work:

```bibtex
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters}, 
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314}, 
}
```

## License

Apache License 2.0s
