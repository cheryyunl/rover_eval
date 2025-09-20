# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset

# Import metric functions
from metric_temporal import evaluate_images as evaluate_temporal
from metric_spatial import evaluate_images as evaluate_spatial  
from metric_quantitative import evaluate_images as evaluate_quantitative
from metric_causal import evaluate_images as evaluate_causal
from metric_synthetic import evaluate_images as evaluate_synthetic
from metric_logical import evaluate_images as evaluate_logical

# Hugging Face dataset
DATASET_NAME = "cheryyunl/ROVER-Gen"
VORTEX_GEN_DIR = "/code/gen_banana"

# Metric mapping
REASONING_EVALUATORS = {
    "temporal": evaluate_temporal,
    "spatial": evaluate_spatial,
    "quantitative": evaluate_quantitative, 
    "causal": evaluate_causal,
    "synthetic": evaluate_synthetic,
    "logical": evaluate_logical,
    "mathematical": evaluate_logical,  # Map mathematical to logical
    "abstract": evaluate_logical       # Map abstract to logical
}

METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment", "visual_consistency", "image_quality"]


def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file"""
    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
        data = {"key": key, "result": result}
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def process_task_evaluation(task, vortex_data, metrics, api_key, output_jsonl_path):
    """Process single task evaluation"""
    try:
        task_id = task["id"]
        reasoning_type = task["reasoning_type"]
        
        # Get the appropriate evaluator
        evaluator = REASONING_EVALUATORS.get(reasoning_type)
        if not evaluator:
            logging.error(f"No evaluator found for reasoning_type: {reasoning_type}")
            return False
            
        # Run evaluation with unified new signature
        result = evaluator(
            image_id=task_id,
            metrics=metrics,
            vortex_data=vortex_data,
            api_key=api_key
        )
        
        # Save result
        save_result_jsonl(result, task_id, output_jsonl_path)
        return True
        
    except Exception as e:
        logging.error(f"Error processing task {task.get('id', 'unknown')}: {e}")
        return False


def load_huggingface_data():
    """Load data from Hugging Face dataset"""
    try:
        dataset = load_dataset(DATASET_NAME)
        print(f"Loaded dataset {DATASET_NAME}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading Hugging Face dataset {DATASET_NAME}: {e}")
        return None

def convert_hf_to_vortex_format(dataset):
    """Convert Hugging Face dataset to VortexBench format"""
    tasks = []
    
    # Get the train split
    split_data = dataset['train'] if 'train' in dataset else dataset
    
    # Get dimension labels mapping
    dimension_names = split_data.features['dimension'].names
    
    for item in split_data:
        # Extract and process fields
        keywords = item.get('keywords', '')  # This is already a string
        target_description = item.get('target_description', '')
        
        # Convert dimension index to name
        dimension_idx = item.get('dimension')
        dimension = dimension_names[dimension_idx] if dimension_idx is not None else 'unknown'
        
        task = {
            'id': item.get('id'),
            'image_file': item.get('image_file'),
            'dimension': dimension,  # Convert index to name
            'reasoning_type': item.get('reasoning_type'),
            'prompt': item.get('prompt'),
            'target_description': target_description,
            'keywords': keywords,  # Already a string
            'image': item.get('image'),  # PIL Image object
            'target_image': item.get('target_image'),  # PIL Image object (if exists)
        }
        tasks.append(task)
    
    return {'tasks': tasks}

def run_vortex_evaluation(
    output_dir="vortex_results",
    num_workers=10,
    metrics=None,
    api_key=None,
    filter_dimension=None,
    filter_reasoning_type=None
):
    """
    Run VortexBench evaluation using Hugging Face dataset
    
    Args:
        output_dir: Directory to save results
        num_workers: Number of parallel workers
        metrics: List of metrics to evaluate
        api_key: OpenAI API key
        filter_dimension: Filter by dimension (science/humanity/common_sense/logic)
        filter_reasoning_type: Filter by reasoning type (temporal/spatial/quantitative/causal/synthetic)
    """
    metrics = metrics or METRICS
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir, "vortex_metrics.jsonl")
    
    # Load Hugging Face dataset
    dataset = load_huggingface_data()
    if dataset is None:
        return
    
    # Convert to VortexBench format
    vortex_data = convert_hf_to_vortex_format(dataset)
    
    # Filter tasks
    tasks = vortex_data["tasks"]
    if filter_dimension:
        tasks = [t for t in tasks if t.get("dimension") == filter_dimension]
    if filter_reasoning_type:
        tasks = [t for t in tasks if t.get("reasoning_type") == filter_reasoning_type]
    
    print(f"Found {len(tasks)} tasks to evaluate")
    
    # Check which tasks have generated images
    valid_tasks = []
    for task in tasks:
        task_id = task["id"]
        gen_image_path = os.path.join(VORTEX_GEN_DIR, f"gen_{task_id}.png")
        if os.path.exists(gen_image_path):
            valid_tasks.append(task)
        else:
            print(f"Warning: Generated image not found for {task_id}")
    
    print(f"Found {len(valid_tasks)} tasks with generated images")
    
    if not valid_tasks:
        print("No tasks with generated images found. Please run generation first.")
        return
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for task in valid_tasks:
            future = executor.submit(
                process_task_evaluation,
                task, vortex_data, metrics, api_key, output_jsonl_path
            )
            futures.append(future)
        
        # Process results with progress bar
        successful = 0
        failed = 0
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating VortexBench"):
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Future failed: {e}")
                failed += 1
    
    print(f"Evaluation completed: {successful} successful, {failed} failed")
    print(f"Results saved to: {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VortexBench Evaluation")
    parser.add_argument("--output_dir", type=str, default="vortex_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS, help="Metrics to evaluate")
    parser.add_argument("--api_key", type=str, help="[DEPRECATED] API key parameter - Azure credentials are configured in metric files")
    parser.add_argument("--dimension", type=str, choices=["science", "humanity", "common_sense", "logic"], help="Filter by dimension")
    parser.add_argument("--reasoning_type", type=str, choices=["temporal", "spatial", "quantitative", "causal", "synthetic"], help="Filter by reasoning type")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # API key handling (deprecated - now configured in metric files)
    api_key = args.api_key
    if api_key:
        print("Warning: --api_key parameter is deprecated. Azure credentials are configured in metric files.")
    
    run_vortex_evaluation(
        output_dir=args.output_dir,
        num_workers=args.workers,
        metrics=args.metrics,
        api_key=api_key,
        filter_dimension=args.dimension,
        filter_reasoning_type=args.reasoning_type
    )
