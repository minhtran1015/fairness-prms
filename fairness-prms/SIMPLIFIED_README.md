# Simplified Fairness Evaluation

A simplified, standalone script for running fairness evaluation using Process Reward Models (PRMs) on the BBQ dataset.

## Features

✅ **Simple & Standalone**: Single script with no complex dependencies
✅ **Better Error Handling**: Clear error messages and graceful fallbacks  
✅ **Library Compatibility**: Works with multiple versions of datasets library
✅ **Progress Tracking**: Clear logging and progress bars
✅ **Dual GPU Support**: Efficient tensor parallelism for faster inference

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers datasets vllm==0.6.3 tqdm
```

**Note**: Use `datasets==2.14.0` for best compatibility with custom dataset scripts.

### 2. Run Evaluation

```bash
python scripts/run_fairness_eval.py \
    --dataset-config SES \
    --num-samples 50 \
    --num-candidates 8 \
    --tensor-parallel-size 2
```

### 3. View Results

Results are saved to `./fairness_results/`:
- `fairness_eval_results.jsonl`: Detailed results for each example
- `summary_stats.json`: Summary statistics

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `meta-llama/Llama-3.2-1B-Instruct` | Language model to evaluate |
| `--prm` | `zarahall/bias-prm-v3` | Process Reward Model for scoring |
| `--dataset-config` | `SES` | BBQ category (SES, Age, Gender_identity, etc.) |
| `--num-samples` | `50` | Number of examples to evaluate |
| `--num-candidates` | `8` | Number of candidates for Best-of-N sampling |
| `--temperature` | `0.7` | Sampling temperature |
| `--tensor-parallel-size` | `2` | Number of GPUs to use |
| `--output-dir` | `./fairness_results` | Output directory |

## BBQ Dataset Categories

The BBQ (Bias Benchmark for QA) dataset includes multiple bias categories:

- `SES` - Socioeconomic status
- `Age` - Age bias
- `Gender_identity` - Gender identity bias
- `Race_ethnicity` - Race and ethnicity bias
- `Disability_status` - Disability status bias
- `Nationality` - Nationality bias
- `Physical_appearance` - Physical appearance bias
- `Religion` - Religious bias
- `Sexual_orientation` - Sexual orientation bias

## How It Works

1. **Load Dataset**: Loads BBQ dataset with the specified category
2. **Filter Examples**: Filters for ambiguous context examples
3. **Generate Candidates**: Uses Best-of-N sampling to generate multiple responses
4. **Score with PRM**: Scores each candidate using the fairness-aware PRM
5. **Select Best**: Selects the response with the highest fairness score
6. **Save Results**: Saves detailed results and summary statistics

## Example Output

```json
{
  "example_id": 1234,
  "category": "SES",
  "question": "Who is more likely to...",
  "candidates": ["Response 1", "Response 2", ...],
  "scores": [0.85, 0.72, ...],
  "best_response": "Response 1",
  "best_score": 0.85
}
```

## Troubleshooting

### Dataset Loading Issues

If you get errors loading the BBQ dataset:

```bash
# Install specific datasets version
pip install datasets==2.14.0
```

### GPU Memory Issues

If you run out of GPU memory:

```bash
# Reduce batch size or number of candidates
python scripts/run_fairness_eval.py \
    --num-candidates 4 \
    --tensor-parallel-size 1
```

### vLLM Issues

Make sure you have vLLM 0.6.3 installed:

```bash
pip install vllm==0.6.3
```

## Advanced Usage

### Custom Model

```bash
python scripts/run_fairness_eval.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prm zarahall/bias-prm-v3
```

### Evaluate Multiple Categories

```bash
for category in SES Age Gender_identity Race_ethnicity; do
    python scripts/run_fairness_eval.py \
        --dataset-config $category \
        --output-dir "./results/$category"
done
```

## Performance Tips

- **Dual T4 GPUs**: Use `--tensor-parallel-size 2` for ~2x speedup
- **Larger Models**: Increase `--tensor-parallel-size` for bigger models
- **Fast Testing**: Use `--num-samples 10` for quick tests

## Citation

If you use this code, please cite the original BBQ paper:

```bibtex
@inproceedings{parrish2022bbq,
  title={BBQ: A hand-built bias benchmark for question answering},
  author={Parrish, Alicia and Chen, Angelica and Nangia, Nikita and Padmakumar, Vishakh and Phang, Jason and Thompson, Jana and Htut, Phu Mon and Bowman, Samuel},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
```
