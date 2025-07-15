import math
from typing import Literal

from datasets import Dataset
from tqdm import tqdm
import torch

from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
    aggregate_scores,
    find_answer_with_largest_sum,
)


def compute_weighted_preds_multi_temp(x, n, config=None, temps=[0.01, 0.1, 0.2, 0.4]):
    """
    Compute weighted predictions for multiple temperatures and add them to the output dict.
    """
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    print(f"DEBUG: preds={preds}, scores={scores}")
    # Ensure all scores are numeric values for weighting calculations
    scores = [float(s) if not isinstance(s, (int, float)) else s for s in scores]
    results = {}
    for temp in temps:
        key = f"pred_weighted_temp{temp}@{n}"
        pred = "\\boxed{" + find_answer_with_largest_sum(preds, scores, is_log_score=False, temperature=temp) + "}"
        print(f"DEBUG: temp={temp}, pred={pred}")
        results[key] = pred
    return results


def score(dataset: Dataset, config: Config) -> Dataset:
    """
    Post-process the scores that were already computed in best_of_n.py.
    The dataset is expected to already have the 'scores' column.
    """
    # First, aggregate the scores for each completion, handling both step-level and single scores
    dataset = dataset.map(
        lambda x: {
            "agg_scores": [
                # Check if each score is a list (BiasDetectionPRM) or a single value (OutcomeDetectionPRM)
                aggregate_scores(s, config.search.agg_strategy) if isinstance(s, list) else s 
                for s in x["scores"]
            ]
        }
    )
    
    # Process subsets for evaluation metrics
    subsets = [2**i for i in range(config.search.n) if 2**i <= config.search.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.output.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n, "config": config},
            num_proc=config.output.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_preds_multi_temp,
            fn_kwargs={"n": n, "config": config, "temps": [0.01, 0.2, 0.4, 0.8]},
            num_proc=config.output.num_proc,
            desc=f"Compute weighted preds (multi-temp) {n}",
            load_from_cache_file=False,
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.output.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.output.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset
