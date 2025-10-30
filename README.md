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
  - 11 bias categories: Age, Disability_status, Gender_identity, Nationality, Physical_appearance, Race_ethnicity, Race_x_gender, Race_x_SES, Religion, SES, Sexual_orientation
  - Automated evaluation across all categories with multi-temperature testing
- **COMPAS**: Recidivism risk assessment with racial fairness considerations
- **Bias in Bios**: Occupation classification from biographical text
- **Civil Comments**: Content moderation with fairness constraints

## Key Features

### 🔬 Automated Evaluation Pipeline

The `fairness-reward-model.ipynb` notebook provides:

- **Complete bias evaluation** across all 11 BBQ categories
- **Multi-temperature testing** (0.01, 0.2, 0.4, 0.8) for robustness analysis
- **GPU-optimized setup** for 2x Tesla T4 configurations
- **Progress tracking** with checkpointing and time management
- **Built-in analysis** with accuracy calculation and result merging

### 🖥️ Hardware Optimization

- **Dual GPU support** with tensor parallelism for 2x inference speedup
- **Automatic GPU detection** and optimal configuration
- **Memory management** optimized for Tesla T4 GPUs
- **FlashAttention/XFormers** backend selection based on compute capability

### 🛠️ Robust Data Handling

- **Direct dataset downloads** bypassing `datasets` library cache issues
- **JSONL format support** for reliable data loading
- **Error handling** with multiple fallback methods for model loading

## Quick Start

### 🚀 Recommended: Use the Interactive Notebook

The easiest way to run fairness evaluations is using the comprehensive notebook:

```bash
# Clone and navigate to the repository
git clone https://github.com/minhtran1015/fairness-prms
cd fairness-prms

# Open the notebook (works on Kaggle, Colab, or local)
fairness-reward-model.ipynb
```

The notebook provides:

- ✅ **Automated evaluation** across all bias categories
- ✅ **Multi-temperature testing** (0.01, 0.2, 0.4, 0.8)
- ✅ **GPU optimization** for 2x Tesla T4 setup
- ✅ **Progress tracking** and error handling
- ✅ **Built-in analysis** and accuracy calculation

### Alternative: Command Line Scripts

#### Simple Evaluation (Single Dataset)

```bash
python scripts/run_fairness_eval.py \
    --dataset-config SES \
    --num-samples 50 \
    --num-candidates 8 \
    --tensor-parallel-size 2
```

#### Advanced Configuration (YAML-based)

```bash
python scripts/test_time_compute.py --recipe "llama_best_of_n"
```

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

```bash
fairness-prms/
├── fairness-reward-model.ipynb    # 🆕 Main evaluation notebook
├── scripts/
│   ├── run_fairness_eval.py       # Simplified evaluation script
│   └── test_time_compute.py       # Advanced YAML-based evaluation
├── src/sal/
│   ├── config/                    # Configuration system
│   │   ├── base.py               # Base configuration classes
│   │   ├── prompts.py            # Dataset-specific prompts
│   │   └── loader.py             # Configuration loading
│   ├── models/                   # Reward model implementations
│   │   ├── prm/                  # Process reward models
│   │   │   ├── bias_detection.py # Fairness-aware PRM
│   │   │   └── math_shepherd.py  # Math reasoning PRM
│   │   └── registry.py           # Model registry
│   ├── search/                   # Search algorithms
│   │   ├── best_of_n.py
│   │   ├── beam_search.py
│   │   └── diverse_verifier_tree_search.py
│   └── utils/                    # Utilities for scoring and data processing
├── recipes/                      # YAML configuration recipes
├── fairness_results/             # Evaluation outputs (JSONL + summaries)
└── analysis_output/              # Merged analysis results (CSV/Excel/JSON)
```

## Output Formats

### Evaluation Results (`fairness_results/`)

- **JSONL format**: Detailed results for each example with candidates, scores, and best response
- **Summary statistics**: Average PRM scores, min/max values, and accuracy metrics
- **Organized by category**: Results stored in category-specific directories

### Analysis Results (`analysis_output/`)

- **CSV/Excel/JSON**: Merged results combining original data with evaluation outputs
- **Multi-temperature analysis**: Results across different sampling temperatures
- **Accuracy calculations**: Automated accuracy computation and reporting

### Sample Output Structure

```json
{
  "example_id": 1234,
  "category": "SES",
  "question": "Who uses drugs?",
  "candidates": ["Response 1", "Response 2", ...],
  "scores": [0.85, 0.72, ...],
  "best_response": "Response 1",
  "best_score": 0.85,
  "label": "A"
}
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
