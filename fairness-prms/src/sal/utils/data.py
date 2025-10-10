# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from pathlib import Path
import json
import random
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import (
    get_full_repo_name,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config
logger = logging.getLogger()
def _load_bbq_dataset(config: Config) -> Dataset:
    """Load and prepare the BBQ dataset."""
    dataset = load_dataset("heegyu/bbq", config.dataset.config, split='test', trust_remote_code=True)
    
    # Filter for ambiguous context condition
    dataset = dataset.filter(lambda x: x['context_condition'] == 'ambig')
    
    if config.dataset.config == "Age":
        # Filter for examples with ID greater than 2000 for Age config
        filtered_dataset = dataset.filter(lambda x: x['example_id'] > 2000)
    else:
        # Filter for examples with ID greater than 500 for other configs
        filtered_dataset = dataset.filter(lambda x: x['example_id'] > 500)
        
    # Select 500 examples from the filtered dataset
    filtered_dataset = filtered_dataset.shuffle(seed=45).select(range(min(len(filtered_dataset), 500)))
    return filtered_dataset
def _load_bias_in_bios_dataset(config: Config) -> Dataset:
    """Load and prepare the Bias in Bios dataset with counterfactuals."""
    dataset = load_dataset("LabHC/bias_in_bios", split=config.dataset.split)

    # Use numeric indices or string labels if available
    valid_professions = [13, 19, 26, 21]  # Replace with actual labels if necessary
    dataset = dataset.filter(lambda x: x['profession'] in valid_professions)
    
    # Add original index to track pairs
    dataset = dataset.add_column("original_index", list(range(len(dataset))))
    
    # Create a combined dataset with 100 examples for each profession and gender
    combined_dataset = Dataset.from_dict({"text": [], "gender": [], "profession": [], "hard_text": [], "original_index": []})
    
    for profession in valid_professions:
        # Filter for each gender
        male_examples = dataset.filter(lambda x: x['profession'] == profession and x['gender'] == 0)
        female_examples = dataset.filter(lambda x: x['profession'] == profession and x['gender'] == 1)
        
        # Sample 100 examples from each gender for each profession
        male_sample = male_examples.shuffle(seed=42).select(range(min(len(male_examples), 100)))
        female_sample = female_examples.shuffle(seed=42).select(range(min(len(female_examples), 100)))
        
        # Combine samples
        combined_dataset = concatenate_datasets([combined_dataset, male_sample, female_sample])
    
    logger.info(f"Loaded bias in bios dataset with {len(combined_dataset)} examples across {len(valid_professions)} professions.")
    return combined_dataset

def _load_math_dataset(config: Config) -> Dataset:
    """Load and prepare the MATH-500 dataset."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split='test')
    return dataset.shuffle(seed=42).select(range(100))

def _load_compas_dataset(config: Config) -> Dataset:
    """Load and prepare the COMPAS dataset."""
    dataset = load_dataset("AiresPucrs/COMPAS", split='train')
    
    # Filter for African-American males
    caucasian_females = dataset.filter(lambda x: x['race'] == 'African-American' and x['sex'] == 'Male')
    
    # Filter for Caucasian males
    caucasian_males = dataset.filter(lambda x: x['race'] == 'Caucasian' and x['sex'] == 'Male')
    # Sample 250 from each group with a different seed than 42
    caucasian_females_sample = caucasian_females.shuffle(seed=42).select(range(min(len(caucasian_females), 250)))
    caucasian_sample = caucasian_males.shuffle(seed=42).select(range(min(len(caucasian_males), 250)))
    # Combine the samples
    dataset = concatenate_datasets([caucasian_females_sample, caucasian_sample])
    return dataset

def _load_civilcomments_dataset(config: Config) -> Dataset:
    """Load and prepare the CivilComments dataset."""
    dataset = load_dataset("zarahall/civilcomments", split='train')
    # Filter for demographics
    muslim = dataset.filter(lambda x: x['muslim'] > 0.5 and x['christian'] < 0.5)
    christian = dataset.filter(lambda x: x['muslim'] > 0.5 and x['christianlo'] < 0.5)
    logger.info(f"Heterosexual examples: {len(muslim)}, Homosexual examples: {len(christian)}")
    
    # Sample from each category
    sample_size = 250
    muslim_sample = muslim.shuffle(seed=42).select(
        range(min(len(muslim), sample_size))
    )
    christian_sample = christian.shuffle(seed=42).select(
        range(min(len(christian), sample_size))
    )
    
    # Combine the samples
    return concatenate_datasets([
        muslim_sample,
        christian_sample
    ])
    
def get_dataset(config: Config) -> Dataset:
    """Load a dataset based on the configuration.
    
    Args:
        config: The configuration object containing dataset specifications.
        
    Returns:
        A prepared dataset according to the configuration.
        
    Raises:
        ValueError: If the dataset name is not recognized.
    """
    dataset_loaders = {
        "heegyu/bbq": _load_bbq_dataset,
        "LabHC/bias_in_bios": _load_bias_in_bios_dataset,
        "HuggingFaceH4/MATH-500": _load_math_dataset,
        "zarahall/civilcomments": _load_civilcomments_dataset,
        "AiresPucrs/COMPAS": _load_compas_dataset,
    }
    
    if config.dataset.name in dataset_loaders:
        dataset = dataset_loaders[config.dataset.name](config)
    else:
        raise ValueError(f"Dataset {config.dataset.name} is not recognized.")

    logger.info(f"Final dataset size: {len(dataset)}")
    return dataset

def extract_reasoning_from_completions(
    example: Dict[str, Any], 
    label_id: int, 
    pattern: List[str]
) -> Optional[Dict[str, Any]]:
    """Extract reasoning from completions that match a specific pattern.
    
    Args:
        example: The dataset example containing completions
        label_id: The correct label ID to match
        pattern: List of strings to look for in completions
        
    Returns:
        A dictionary with reasoning information or None if no match found
    """
    if example["label"] != label_id:
        return None
        
    matching_completions = []
    for i, completion in enumerate(example["completions"]):
        if any(p in completion.lower() for p in pattern):
            matching_completions.append({
                "completion": completion,
                "score": example["scores"][i][-1]  # Use final score
            })
    
    if not matching_completions:
        return None
        
    # Sort by score and take highest scoring completion
    best_completion = max(matching_completions, key=lambda x: x["score"])
    return {
        "example_id": example["example_id"],
        "reasoning": best_completion["completion"],
        "score": best_completion["score"]
    }

def extract_tree_of_thought(dataset: Dataset, config: Config) -> None:
    """Extract and save the tree of thought reasoning process for each example.
    
    Args:
        dataset: The dataset containing completions
        config: Configuration with output settings
    """
    reasoning_dataset = []
    
    for example in dataset:
        result = extract_reasoning_from_completions(
            example, 
            label_id=1, 
            pattern=["not enough information", "cannot be determined"]
        )
        if result:
            reasoning_dataset.append(result)
    
    # Save the reasoning dataset if we extracted any examples
    if reasoning_dataset:
        output_path = Path(config.output.output_dir) / "reasoning.jsonl"
        with open(output_path, "w") as f:
            for item in reasoning_dataset:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved reasoning dataset to {output_path}")

def _save_to_local(dataset: Dataset, config: Config) -> None:
    """Save the dataset to a local directory.
    
    Args:
        dataset: The dataset to save
        config: Configuration with output settings
    """
    if config.output.output_dir is None:
        config.output.output_dir = f"data/{config.model.model_path.split('/')[-1]}"
    
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the original dataset
    dataset.to_json(f"{config.output.output_dir}/bon_completions.jsonl", lines=True)
    logger.info(f"Saved completions to {config.output.output_dir}/bon_completions.jsonl")
    
    # Extract and save tree of thought reasoning
    extract_tree_of_thought(dataset, config)

def _save_to_hub(dataset: Dataset, config: Config) -> None:
    """Save the dataset to the Hugging Face Hub."""
    max_attempts = 20

    def make_hf_dataset_name(config):
        dataset = config.dataset.name.replace('/', '_')
        subset = getattr(config.dataset, 'config', None)
        model = getattr(config.model, 'model_name', None) or getattr(config.model, 'model_path', '').split('/')[-1]
        prm_tag = config.model.prm_paths[0].split('/')[-1] if config.model.prm_paths else 'prm'
        math_temp = config.search.math_temperature
        mt_str = f"{math_temp:.2g}"
        parts = [dataset]
        if subset:
            parts.append(subset)
        if model:
            parts.append(model)
        parts.append(f"prm-{prm_tag}")
        parts.append(f"mt-{mt_str}")
        return "_".join(parts)

    for attempt in range(max_attempts):
        try:
            hf_dataset_name = make_hf_dataset_name(config)
            url = dataset.push_to_hub(
                hf_dataset_name,
                revision="main",
                split="train",
                private=config.output.hub_dataset_private,
                commit_message="Add dataset",
            )
            logger.info(f"Pushed dataset to {url}")
            return
        except Exception as e:
            logger.error(f"Error pushing dataset to the Hub (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error(f"Failed to push dataset after {max_attempts} attempts")

def save_dataset(dataset: Dataset, config: Config) -> None:
    """Save the dataset based on the configuration.
    
    Args:
        dataset: The dataset to save
        config: Configuration with output settings
    """
    if config.output.push_to_hub:
        _save_to_hub(dataset, config)
    else:
        _save_to_local(dataset, config)
