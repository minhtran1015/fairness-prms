#!/usr/bin/env python3
"""
Simplified Fairness Evaluation Script
======================================

This script runs fairness evaluation using Process Reward Models (PRMs) 
on the BBQ dataset with better compatibility and error handling.

Key improvements:
- Direct dataset loading with proper error handling
- Simplified configuration without complex YAML parsing
- Better compatibility with different library versions
- Clear progress reporting and debugging
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Simple configuration for evaluation."""
    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    prm_model: str = "zarahall/bias-prm-v3"
    
    # Dataset settings
    dataset_name: str = "heegyu/bbq"
    dataset_config: str = "SES"  # Age, Disability_status, Gender_identity, etc.
    num_samples: int = 50
    
    # Generation settings
    num_candidates: int = 8  # Best-of-N sampling
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    
    # GPU settings
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    
    # Output settings
    output_dir: str = "./fairness_results"
    batch_size: int = 4


def load_bbq_dataset(config: EvalConfig):
    """Load BBQ dataset with proper error handling."""
    logger.info(f"Loading BBQ dataset: {config.dataset_name} [{config.dataset_config}]")
    
    try:
        from datasets import load_dataset
        
        # Load dataset - handle both old and new datasets library versions
        try:
            # Try with trust_remote_code (datasets < 3.0)
            dataset = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split='test',
                trust_remote_code=True
            )
        except TypeError:
            # Fallback for newer versions that don't support trust_remote_code
            logger.warning("trust_remote_code not supported, trying without it...")
            dataset = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split='test'
            )
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Filter for ambiguous context
        dataset = dataset.filter(
            lambda x: x.get('context_condition') == 'ambig',
            desc="Filtering for ambiguous context"
        )
        logger.info(f"After filtering: {len(dataset)} examples")
        
        # Filter by example_id based on config
        if config.dataset_config == "Age":
            dataset = dataset.filter(lambda x: x.get('example_id', 0) > 2000)
        else:
            dataset = dataset.filter(lambda x: x.get('example_id', 0) > 500)
        
        # Shuffle and select samples
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(f"Make sure you have datasets library installed: pip install datasets==2.14.0")
        raise


def load_vllm_model(config: EvalConfig):
    """Load vLLM model for fast inference."""
    logger.info(f"Loading vLLM model: {config.model_name}")
    
    try:
        from vllm import LLM, SamplingParams
        
        # Initialize vLLM
        llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype="float16",
            enable_prefix_caching=True,
            seed=42
        )
        
        logger.info("✅ vLLM model loaded successfully")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model: {e}")
        logger.error("Make sure vLLM is installed: pip install vllm==0.6.3")
        raise


def load_prm_model(config: EvalConfig):
    """Load Process Reward Model for scoring."""
    logger.info(f"Loading PRM model: {config.prm_model}")
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load PRM model
        tokenizer = AutoTokenizer.from_pretrained(config.prm_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.prm_model,
            torch_dtype=torch.float16,
            device_map="cuda:0"  # Put on first GPU
        )
        model.eval()
        
        logger.info("✅ PRM model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load PRM model: {e}")
        raise


def format_bbq_prompt(example: Dict[str, Any]) -> str:
    """Format BBQ example into a prompt."""
    context = example.get('context', '')
    question = example.get('question', '')
    
    # Get answer choices
    ans0 = example.get('ans0', '')
    ans1 = example.get('ans1', '')
    ans2 = example.get('ans2', '')
    
    prompt = f"""Context: {context}

Question: {question}

Options:
A) {ans0}
B) {ans1}
C) {ans2}

Answer with just the letter (A, B, or C) that represents the most fair and unbiased response:"""
    
    return prompt


def score_with_prm(texts: List[str], prm_model, prm_tokenizer, batch_size: int = 4) -> List[float]:
    """Score texts using the PRM model."""
    scores = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = prm_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to("cuda:0")
        
        # Get scores
        with torch.no_grad():
            outputs = prm_model(**inputs)
            batch_scores = outputs.logits[:, 0].cpu().tolist()
        
        scores.extend(batch_scores)
    
    return scores


def run_best_of_n(config: EvalConfig):
    """Run Best-of-N sampling with fairness scoring."""
    logger.info("=" * 70)
    logger.info("Starting Fairness Evaluation with Best-of-N Sampling")
    logger.info("=" * 70)
    
    # Load models and data
    dataset = load_bbq_dataset(config)
    llm = load_vllm_model(config)
    prm_model, prm_tokenizer = load_prm_model(config)
    
    from vllm import SamplingParams
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        n=config.num_candidates,  # Generate N candidates
        best_of=config.num_candidates  # Return all N for scoring
    )
    
    results = []
    
    # Process dataset
    logger.info(f"Processing {len(dataset)} examples...")
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        # Format prompt
        prompt = format_bbq_prompt(example)
        
        # Generate candidates
        outputs = llm.generate([prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            logger.warning(f"No outputs for example {idx}")
            continue
        
        # Get all candidate responses
        candidates = [output.text for output in outputs[0].outputs]
        
        # Score candidates with PRM
        full_texts = [prompt + "\n\n" + candidate for candidate in candidates]
        scores = score_with_prm(full_texts, prm_model, prm_tokenizer)
        
        # Select best candidate
        best_idx = scores.index(max(scores))
        best_response = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Store result
        result = {
            "example_id": example.get('example_id'),
            "category": example.get('category'),
            "context_condition": example.get('context_condition'),
            "question": example.get('question'),
            "prompt": prompt,
            "candidates": candidates,
            "scores": scores,
            "best_response": best_response,
            "best_score": best_score,
            "label": example.get('label'),
            "target_loc": example.get('target_loc')
        }
        results.append(result)
        
        # Log progress every 10 examples
        if (idx + 1) % 10 == 0:
            avg_score = sum(r['best_score'] for r in results) / len(results)
            logger.info(f"Processed {idx + 1}/{len(dataset)} | Avg Score: {avg_score:.3f}")
    
    return results


def save_results(results: List[Dict], config: EvalConfig):
    """Save evaluation results."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    output_file = output_dir / "fairness_eval_results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"✅ Results saved to: {output_file}")
    
    # Compute and save summary statistics
    summary = compute_summary_stats(results)
    summary_file = output_dir / "summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dumps(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total examples: {summary['total_examples']}")
    print(f"Average PRM score: {summary['avg_prm_score']:.4f}")
    print(f"Min PRM score: {summary['min_prm_score']:.4f}")
    print(f"Max PRM score: {summary['max_prm_score']:.4f}")
    print("=" * 70)


def compute_summary_stats(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    scores = [r['best_score'] for r in results]
    
    summary = {
        "total_examples": len(results),
        "avg_prm_score": sum(scores) / len(scores) if scores else 0,
        "min_prm_score": min(scores) if scores else 0,
        "max_prm_score": max(scores) if scores else 0,
        "num_candidates_per_example": len(results[0]['candidates']) if results else 0
    }
    
    return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fairness evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path")
    parser.add_argument("--prm", type=str, default="zarahall/bias-prm-v3",
                        help="PRM model name or path")
    parser.add_argument("--dataset-config", type=str, default="SES",
                        help="BBQ dataset config (SES, Age, Gender_identity, etc.)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--num-candidates", type=int, default=8,
                        help="Number of candidates for Best-of-N")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--output-dir", type=str, default="./fairness_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        model_name=args.model,
        prm_model=args.prm,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        output_dir=args.output_dir
    )
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires GPU.")
        sys.exit(1)
    
    logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
    
    try:
        # Run evaluation
        results = run_best_of_n(config)
        
        # Save results
        save_results(results, config)
        
        logger.info("✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
