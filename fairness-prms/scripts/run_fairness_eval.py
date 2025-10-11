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
    tensor_parallel_size: int = 2  # Use 2 GPUs (Tesla T4)
    gpu_memory_utilization: float = 0.90  # T4 has 16GB, can use more
    
    # Output settings
    output_dir: str = "./fairness_results"
    batch_size: int = 8  # Increase batch size for dual GPUs


def load_bbq_dataset(config: EvalConfig):
    """Load BBQ dataset directly from Hugging Face without datasets library cache issues."""
    logger.info(f"Loading BBQ dataset: {config.dataset_name} [{config.dataset_config}]")
    
    try:
        import json
        from huggingface_hub import hf_hub_download
        import pandas as pd
        
        logger.info("Downloading dataset directly from Hugging Face Hub...")
        
        # Download the JSONL file directly for the specific config
        # BBQ dataset structure: data/{config}.jsonl
        file_path = hf_hub_download(
            repo_id=config.dataset_name,
            filename=f"data/{config.dataset_config}.jsonl",
            repo_type="dataset"
        )
        logger.info(f"Downloaded file: {file_path}")
        
        # Load the JSONL file
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} examples")
        
        # Filter for ambiguous context
        df = df[df['context_condition'] == 'ambig']
        logger.info(f"After filtering for ambiguous context: {len(df)} examples")
        
        # Filter by example_id based on config
        if config.dataset_config == "Age":
            df = df[df['example_id'] > 2000]
        else:
            df = df[df['example_id'] > 500]
        logger.info(f"After filtering by example_id: {len(df)} examples")
        
        # Shuffle and select samples
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.head(min(len(df), config.num_samples))
        
        # Convert to list of dicts (similar to datasets format)
        dataset = df.to_dict('records')
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(f"Make sure you have pandas and huggingface_hub installed")
        raise


def load_vllm_model(config: EvalConfig):
    """Load vLLM model for fast inference."""
    logger.info(f"Loading vLLM model: {config.model_name}")
    
    try:
        from vllm import LLM, SamplingParams
        import torch
        import os
        
        # Check GPU compute capability and configure accordingly
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            compute_cap = torch.cuda.get_device_capability(0)
            logger.info(f"🖥️  Detected {num_gpus} GPU(s)")
            logger.info(f"🔧 GPU 0 Compute Capability: {compute_cap}")
            
            # Tesla T4 has compute capability 7.5 - excellent support
            if compute_cap[0] >= 7:
                logger.info("✅ Modern GPU detected (Tesla T4 or newer)")
                logger.info("✅ Using optimized settings with FlashAttention")
                enforce_eager = False  # Can use CUDA graphs for better performance
                enable_prefix_caching = True  # Enable for better efficiency
                disable_custom_all_reduce = False  # Enable for multi-GPU
                max_model_len = None  # Use model's default (up to 131k for Llama 3.2)
            # P100 has compute capability 6.0 - needs compatibility mode
            elif compute_cap[0] < 7:
                logger.warning("⚠️  Older GPU detected (P100 or similar)")
                logger.warning("⚠️  Using compatibility mode with TORCH_SDPA attention")
                os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
                enforce_eager = True
                enable_prefix_caching = False
                disable_custom_all_reduce = True
                max_model_len = 4096
            else:
                enforce_eager = False
                enable_prefix_caching = True
                disable_custom_all_reduce = False
                max_model_len = None
        else:
            logger.error("❌ No CUDA GPU detected")
            raise RuntimeError("CUDA GPU required for vLLM")
        
        # Initialize vLLM with optimal settings for detected GPU
        vllm_kwargs = {
            "model": config.model_name,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": "float16",
            "enable_prefix_caching": enable_prefix_caching,
            "seed": 42,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
        }
        
        # Add optional parameters
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len
        if disable_custom_all_reduce:
            vllm_kwargs["disable_custom_all_reduce"] = disable_custom_all_reduce
        if config.tensor_parallel_size > 1:
            vllm_kwargs["distributed_executor_backend"] = "mp"
        
        logger.info(f"🚀 Initializing vLLM with: {vllm_kwargs}")
        llm = LLM(**vllm_kwargs)
        
        logger.info("✅ vLLM model loaded successfully")
        return llm
        
    except Exception as e:
        logger.error(f"❌ Failed to load vLLM model: {e}")
        logger.error("Make sure vLLM is installed: pip install vllm")
        raise


def load_prm_model(config: EvalConfig):
    """Load Process Reward Model for scoring."""
    logger.info(f"Loading PRM model: {config.prm_model}")
    
    # NUCLEAR OPTION: Patch at the source - fix the config BEFORE loading model
    logger.info("Applying pre-emptive config fix...")
    try:
        from transformers import AutoConfig
        
        # Load config and fix it BEFORE it's used in model initialization
        logger.info(f"Pre-loading config from {config.prm_model}...")
        model_config = AutoConfig.from_pretrained(config.prm_model)
        
        # Fix ALL None values that could cause issues in post_init
        fixed_attrs = []
        if hasattr(model_config, 'fsdp'):
            if model_config.fsdp is None:
                model_config.fsdp = ""
                fixed_attrs.append('fsdp')
        
        if hasattr(model_config, 'fsdp_config'):
            if model_config.fsdp_config is None:
                model_config.fsdp_config = {}
                fixed_attrs.append('fsdp_config')
        
        # Also check for other parallel-related attributes
        parallel_attrs = ['fsdp', 'fsdp_config', 'deepspeed', 'gradient_accumulation_steps']
        for attr in parallel_attrs:
            if hasattr(model_config, attr) and getattr(model_config, attr) is None:
                if attr == 'fsdp_config':
                    setattr(model_config, attr, {})
                else:
                    setattr(model_config, attr, "")
                if attr not in fixed_attrs:
                    fixed_attrs.append(attr)
        
        if fixed_attrs:
            logger.info(f"✅ Pre-emptively fixed config attributes: {fixed_attrs}")
        else:
            logger.info("✅ Config check passed (no None values found)")
            
    except Exception as patch_error:
        logger.warning(f"Could not pre-load config: {patch_error}")
        model_config = None
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(config.prm_model)
        
        # Use the pre-fixed config from above (model_config should be defined)
        if model_config is None:
            logger.warning("Pre-fixed config not available, loading fresh config...")
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(config.prm_model)
            # Emergency fix
            if hasattr(model_config, 'fsdp') and model_config.fsdp is None:
                model_config.fsdp = ""
        
        logger.info(f"Using config: {type(model_config).__name__}")
        
        # HARDCODED: Always use cuda:1 for PRM to avoid any conditional logic issues
        # vLLM uses tensor parallelism across GPUs 0,1, PRM goes on GPU 1
        prm_device = "cuda:1"
        logger.info(f"Loading PRM on {prm_device} (hardcoded for dual-GPU setup)")
        
        # Clear GPU cache before loading PRM to free up memory
        logger.info("Clearing GPU cache before loading PRM...")
        torch.cuda.empty_cache()
        
        # Log memory status
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved, {mem_total:.2f}GB total")
        
        # Load PRM model with pre-fixed config
        logger.info("Loading PRM model weights with pre-fixed config...")
        # Use device_map as dict to avoid tensor parallelism issues
        device_map_dict = {"": prm_device}  # Empty key means all layers go to this device
        model = AutoModelForSequenceClassification.from_pretrained(
            config.prm_model,
            config=model_config,  # Use pre-fixed config
            torch_dtype=torch.float16,
            device_map=device_map_dict,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        logger.info(f"✅ PRM model loaded successfully on {prm_device}")
        logger.info(f"   Model memory usage: ~{sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load PRM model (primary method): {e}")
        logger.error(f"   Model: {config.prm_model}")
        logger.error(f"   Error type: {type(e).__name__}")
        
        # FALLBACK: Try the MOST AGGRESSIVE approach - skip post_init entirely
        try:
            logger.info("🔧 Attempting AGGRESSIVE fallback: Bypassing post_init...")
            from transformers import AutoTokenizer, LlamaForSequenceClassification, LlamaConfig
            import transformers.modeling_utils
            
            tokenizer = AutoTokenizer.from_pretrained(config.prm_model)
            
            # NUCLEAR OPTION: Replace post_init with a no-op to bypass the buggy check
            logger.info("Replacing post_init with safe version...")
            original_post_init = transformers.modeling_utils.PreTrainedModel.post_init
            
            def safe_post_init(self):
                """Safe post_init that skips the buggy parallel style validation."""
                # Just do weight tying, skip the buggy fsdp validation
                if hasattr(self, 'tie_weights'):
                    try:
                        self.tie_weights()
                    except:
                        pass
                # Skip all the problematic validation code
                logger.debug("post_init bypassed successfully")
            
            # Temporarily replace post_init
            transformers.modeling_utils.PreTrainedModel.post_init = safe_post_init
            
            try:
                # Load config and model with bypassed post_init
                logger.info("Loading config as LlamaConfig...")
                raw_config = LlamaConfig.from_pretrained(config.prm_model)
                
                # Still try to fix config attributes just in case
                if hasattr(raw_config, 'fsdp') and raw_config.fsdp is None:
                    raw_config.fsdp = ""
                if hasattr(raw_config, 'fsdp_config') and raw_config.fsdp_config is None:
                    raw_config.fsdp_config = {}
                
                # HARDCODED: Always use cuda:1 for PRM
                target_device = "cuda:1"
                logger.info(f"Loading model directly to {target_device} (hardcoded for dual-GPU setup)...")
                
                # Load model with device_map as dict to avoid tensor parallelism issues
                device_map_dict = {"": target_device}  # Empty key means all layers go to this device
                model = LlamaForSequenceClassification.from_pretrained(
                    config.prm_model,
                    config=raw_config,
                    torch_dtype=torch.float16,
                    device_map=device_map_dict,  # Use dict format!
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.eval()
                
                logger.info(f"✅ PRM model loaded successfully using aggressive fallback on {target_device}")
                return model, tokenizer
                
            finally:
                # Restore original post_init
                transformers.modeling_utils.PreTrainedModel.post_init = original_post_init
                logger.debug("Restored original post_init")
            
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")
            logger.error(f"   Error type: {type(e2).__name__}")
            logger.error("Please check if the model exists and is accessible")
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
        json.dump(summary, f, indent=2)
    
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
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
