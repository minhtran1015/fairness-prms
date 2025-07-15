import math
import random
import signal
from collections import defaultdict
from multiprocessing import Manager
from typing import Any, Dict, List, Literal

import numpy as np
from latex2sympy2 import latex2sympy
from sympy import latex, simplify

from .qwen_math_parser import extract_answer, strip_string


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last", "geometric_mean", "log_sum", "normalized_product"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "geometric_mean":
        return math.prod(scores) ** (1 / len(scores)) if scores else 0
    elif agg_strategy == "log_sum":
        if not scores or any(s <= 0 for s in scores):
            return 0
        return sum(math.log(s) for s in scores) / len(scores)
    elif agg_strategy == "normalized_product":
        normalization_factor = 1 / len(scores) if scores else 0
        return math.prod(scores) ** normalization_factor
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


# Timeout exception
class TimeoutException(Exception):
    pass


# Signal handler for timeout
def timeout_handler(signum, frame):
    raise TimeoutException


manager = Manager()
shared_cache = manager.dict()


def memoized_canonical_form(expression: str, timeout_seconds: int = 3) -> str:
    """
    Compute a canonical form for a multiple choice answer.
    Uses a shared cache across processes for memoization.

    Args:
        expression (str): A multiple choice answer string.
        timeout_seconds (int): Unused parameter kept for API compatibility.

    Returns:
        str: The canonical form of the answer (stripped).
    """
    # Check if the result is already cached
    if expression in shared_cache:
        return shared_cache[expression]

    # Simple normalization for multiple choice answers
    canonical_form = strip_string(expression)
    shared_cache[expression] = canonical_form
    return canonical_form


def subsample_completions(x: Dict[str, List[Any]], n: int) -> Dict[str, List[Any]]:
    completions = x["completions"]
    agg_scores = x["agg_scores"]
    if len(completions) != len(agg_scores):
        raise ValueError(
            f"The number of completions and agg_scores should be the same. Got {len(completions)} completions and {len(agg_scores)} agg_scores."
        )

    # Take the first n samples, as the completions are ordered in groups of size m e.g [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
    # We need to ensure these groups are not broken up in order to have a valid comparison at smaller n
    return {
        f"completions@{n}": completions[:n],
        f"agg_scores@{n}": agg_scores[:n],
    }

def extract_completion_answers(
    x: Dict[str, List[Any]], n: int | None = None, config=None
) -> Dict[str, List[str]]:
    # Get data_name from config, default to "math500" if not specified
    data_name = config.search.data_name if config is not None else "math500"
    
    if n is None:
        result = {"preds": [extract_answer(p, data_name) for p in x["completions"]]}
        return result
    else:
        result = {
            f"preds@{n}": [extract_answer(p, data_name) for p in x[f"completions@{n}"]]
        }
        return result

def compute_naive_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    
    # Ensure all scores are numeric values for sorting
    scores = [float(s) if not isinstance(s, (int, float)) else s for s in scores]
    
    preds = [
        (p, s) for p, s in sorted(zip(preds, scores), key=lambda x: x[1], reverse=True)
    ]
    
    result = {f"pred_naive@{n}": "\\boxed{" + preds[0][0] + "}"}
    return result


def compute_weighted_pred(x: Dict[str, List[Any]], n: int, config=None) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    
    # Ensure all scores are numeric values for weighting calculations
    scores = [float(s) if not isinstance(s, (int, float)) else s for s in scores]
    
    # Use config.search.math_temperature if config is provided, otherwise use default
    temperature = config.search.math_temperature if config is not None else 0.5
    
    return {
        f"pred_weighted@{n}": "\\boxed{"
        + find_answer_with_largest_sum(preds, scores, is_log_score=False, temperature=temperature)
        + "}"
    }


def compute_maj_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, List[str]]:
    preds = x[f"preds@{n}"]
    return {f"pred_maj@{n}": "\\boxed{" + find_majority_answer(preds) + "}"}
def find_answer_with_largest_sum(answers: List[str], scores: List[float], is_log_score: bool = False, temperature: float = 0.5) -> str:
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    # Ensure all scores are valid numeric values
    scores = [float(s) if not isinstance(s, (int, float)) else s for s in scores]
    
    # 1) Compute a single global softmax over all scores.
    global_scores_array = np.array(scores)
    # Apply temperature scaling (lower temperature = sharper distribution)
    scaled_scores = global_scores_array / temperature
    # Subtract the max for numerical stability
    max_score = np.max(scaled_scores)
    global_weights = np.exp(scaled_scores - max_score)
    sum_weights = np.sum(global_weights)
    # Handle case where all weights are effectively zero
    if sum_weights == 0:
        # Just use the raw scores as weights
        global_weights = global_scores_array / np.sum(global_scores_array) if np.sum(global_scores_array) > 0 else np.ones_like(global_scores_array) / len(global_scores_array)
    else:
        global_weights /= sum_weights

    # 2) Group answers by canonical form, summing up the global softmax
    #    probabilities for each group.
    canonical_groups = defaultdict(float)
    canonical_to_original = {}
    for i, answer in enumerate(answers):
        canonical_form = memoized_canonical_form(answer)
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer
        # Just add this answer's softmax probability to the group
        canonical_groups[canonical_form] += global_weights[i]

    # 3) Pick the canonical form with the largest total softmax probability
    max_canonical = max(canonical_groups, key=canonical_groups.get)
    return canonical_to_original[max_canonical]

def find_majority_answer(answers: List[str]) -> str:
    """
    Groups answers based on their canonical forms and finds the group with the largest number of elements.
    In case of a tie, returns the first occurring group with the largest size.

    Args:
        answers (list of str): A list of strings to be grouped.

    Returns:
        str: The string representing the group with the largest number of elements.

    Example:
        answers = ["a", "b", "a", "c"]
        result = find_majority_answer(answers)
        # result would be "a" since "a" appears most frequently.
    """
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    # Group answers using canonical forms
    canonical_groups = defaultdict(int)  # Count occurrences for each canonical form
    canonical_to_original = {}  # Map canonical form back to an original answer

    for answer in answers:
        # Compute the canonical form
        canonical_form = memoized_canonical_form(answer)

        # Increment count for the canonical form
        canonical_groups[canonical_form] += 1

        # Track the original answer for this canonical form
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # Find the canonical form with the largest count
    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            # Return the first occurring group in case of a tie
            return canonical_to_original[canonical_form]


def pass_at_k(n: int, c: int, k: int) -> float:
    """A numerically stable method for calculating an unbiased estimate of pass@k.

    Taken from OpenAI's Codex paper: https://arxiv.org/abs/2107.03374

    Args:
        n (`int`): total number of samples
        c (`int`): number of correct samples
        k (`int`): k in pass@$k$

    Returns:
        `float`: an unbiased estimate of pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(x, k):
    """
    Computes pass@k for predictions, using canonical forms to group and compare answers.

    Args:
        x (dict): A dictionary containing "preds" (list of predictions) and "answer" (correct answer).
        k (int): The cutoff for pass@k.

    Returns:
        dict: A dictionary containing pass@k results.
    """
    n = len(x["preds"])
    if n == 0:
        raise ValueError("No predictions found")
    if x["answer"] == "":
        raise ValueError("Answer is empty")

    # Compute the canonical form of the correct answer
    canonical_answer = memoized_canonical_form(x["answer"])

    # Compute the count of predictions matching the canonical answer
    c = sum(memoized_canonical_form(pred) == canonical_answer for pred in x["preds"])

    # Calculate pass@k
    return {f"pass@{k}": pass_at_k(n, c, k)}


def compute_level(
    x, metric: Literal["mean_score", "pass@1"], name: str, quintiles: List[float]
) -> Dict[str, int]:
    """Computes the difficulty level (1-5) of a problem based on the given metric and quintiles.

    Easier problems have a a higher metric value, so the levels are reversed (1 is the easiest, 5 is the hardest)."""
    if x[metric] < quintiles[0]:
        return {f"level_{name}": 5}
    elif x[metric] < quintiles[1]:
        return {f"level_{name}": 4}
    elif x[metric] < quintiles[2]:
        return {f"level_{name}": 3}
    elif x[metric] < quintiles[3]:
        return {f"level_{name}": 2}
    else:
        return {f"level_{name}": 1}
