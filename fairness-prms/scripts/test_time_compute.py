#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
import os
from pathlib import Path

import torch
from vllm import LLM
from huggingface_hub import login

from sal.config.loader import load_config

from sal.config import Config
from sal.models import load_prm
# from sal.search import beam_search, best_of_n, dvts
from sal.search import best_of_n
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

APPROACHES = {
    # "beam_search": beam_search,
    # "dvts": dvts,
    "best_of_n": best_of_n
}

def load_legacy_config(config_path: str) -> Config:
    """Load configuration from legacy path format (model/approach)."""
    # Extract model and approach from path
    parts = Path(config_path).parts
    if len(parts) >= 2:
        model_name = parts[-2]  # e.g., "Llama-3.2-1B-Instruct"
        approach = Path(parts[-1]).stem  # e.g., "best_of_n" from "best_of_n.yaml"
        
        # Load as recipe if exists, otherwise fall back to direct YAML loading
        try:
            return load_config(recipe=f"{model_name}/{approach}")
        except ValueError:
            return load_config(config_path=config_path)
    return load_config(config_path=config_path)

def main():
    parser = H4ArgumentParser(description="Run test time computation with configuration")
    parser.add_argument("config_path", type=str, help="Path to configuration file or recipe")
    parser.add_argument("--recipe", type=str, help="Optional recipe name to use as base", default=None)
    parser.add_argument("--gpu", type=str, help="Comma-separated list of GPU device numbers to use (e.g., '0,1,2')", default="0")
    args, unknown_args = parser.parse_known_args()
    
    # Convert unknown args to override dict
    overrides = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                key, value = key.split("=", 1)
            else:
                value = "true"
            overrides[key] = value
    
    # Load configuration
    if args.recipe:
        config = load_config(recipe=args.recipe, config_path=args.config_path, **overrides)
    else:
        config = load_legacy_config(args.config_path)
    
    approach_fn = APPROACHES[config.search.approach]
    torch.cuda.empty_cache()

    
    # Parse the number of GPUs to use for tensor parallelism
    gpu_ids = args.gpu.split(',')
    num_gpus = torch.cuda.device_count()
    
    # Initialize models
    llm = LLM(
        model=config.model.model_path,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.search.seed,
        tensor_parallel_size=num_gpus,  # Using multiple GPUs based on input
        dtype="float16",  # <-- Add this line!
    )
    prm = load_prm(config)
    # Process dataset
    dataset = get_dataset(config)
    logger.info(f"Dataset size: {len(dataset)}")
    
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
        cache_file_name=f"/tmp/{config.model.model_path.split('/')[-1]}_{config.search.approach}_cache"
    )
    
    dataset = score(dataset, config)
    save_dataset(dataset, config)
    logger.info("Done 🔥!")

if __name__ == "__main__":
    main()