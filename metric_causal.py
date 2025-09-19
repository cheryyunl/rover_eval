# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import base64
import time
import re
import logging
import requests
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import (
    prompt_reasoning_process_causal,
    prompt_reasoning_visual_causal,
    prompt_reasoning_alignment,
    prompt_visual_consistency,
    prompt_image_quality,
)
import threading

lock = threading.Lock()  # Thread-safe file writing lock

# Import configuration
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

# Define metrics for all reasoning types
METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment", "visual_consistency", "image_quality"]

VORTEX_GEN_DIR = "/code/VortexBench/gen_banana"

def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file with thread lock"""
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {"key": key, "result": result}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys_with_missing_metrics(jsonl_path, metrics, expected_keys_map):
    """Load processed image IDs and return missing metrics for each key"""
    key_missing_metrics = {}  # key -> list of missing metrics
    fully_completed_keys = set()  # keys that have all metrics completed
    
    if os.path.exists(jsonl_path):
        # First, collect all results for each key
        key_results = {}  # key -> merged result dict
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = data["key"]
                    result = data["result"]
                    
                    if key not in key_results:
                        key_results[key] = {}
                    key_results[key].update(result)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Check which metrics are missing for each key
        for key in expected_keys_map:
            if key in key_results:
                missing_metrics = []
                for metric in metrics:
                    # Check if this metric has valid scores
                    if metric == "reasoning_process":
                        if f"reasoning_process_score" not in key_results[key] or key_results[key][f"reasoning_process_score"] is None:
                            missing_metrics.append(metric)
                    elif metric == "reasoning_visual":
                        if f"reasoning_visual_score" not in key_results[key] or key_results[key][f"reasoning_visual_score"] is None:
                            missing_metrics.append(metric)
                    elif metric == "reasoning_alignment":
                        if f"reasoning_alignment_score" not in key_results[key] or key_results[key][f"reasoning_alignment_score"] is None:
                            missing_metrics.append(metric)
                    elif metric == "visual_consistency":
                        if f"visual_consistency_score" not in key_results[key] or key_results[key][f"visual_consistency_score"] is None:
                            missing_metrics.append(metric)
                    elif metric == "image_quality":
                        if f"image_quality_score" not in key_results[key] or key_results[key][f"image_quality_score"] is None:
                            missing_metrics.append(metric)
                
                if missing_metrics:
                    key_missing_metrics[key] = missing_metrics
                else:
                    fully_completed_keys.add(key)
            else:
                key_missing_metrics[key] = metrics[:]  # All metrics missing
    else:
        # File doesn't exist, all keys need all metrics
        key_missing_metrics = {key: metrics[:] for key in expected_keys_map}
    
    return key_missing_metrics, fully_completed_keys

def encode_image_to_base64(image_input):
    """Encode image to base64 string
    
    Args:
        image_input: Can be a file path (string) or PIL Image object
    """
    try:
        if isinstance(image_input, str):
            # It's a file path
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # It's a PIL Image object
            buffer = BytesIO()
            image_input.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_input}: {e}")
        return None

def evaluate_with_gpt(prompt, original_base64=None, edited_base64=None, target_base64=None, max_retries=5):
    """Evaluate images using Azure GPT-4o with intelligent retry mechanism"""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if original_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{original_base64}"}
        })
    
    if edited_base64:
        messages[0]["content"].append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{edited_base64}"}
        })
    
    if target_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{target_base64}"}
        })

    for attempt in range(max_retries):
        try:
            response = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=1000,
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            if content and content.strip():
                # Quick validation - check if response looks like valid JSON
                content = content.strip()
                if (content.startswith('{') and content.endswith('}')) or (content.startswith('{{') and content.endswith('}}')):
                    try:
                        # Handle double braces from prompt format (same as extract_json_field)
                        test_content = content
                        if test_content.startswith('{{') and test_content.endswith('}}'):
                            test_content = test_content[1:-1]  # Remove outer braces
                        
                        json.loads(test_content)
                        time.sleep(1)  # Rate limiting
                        return content  # Return original content for extract_score_and_reason to handle
                    except json.JSONDecodeError:
                        logging.warning(f"Attempt {attempt + 1}: Invalid JSON response, retrying...")
                else:
                    logging.warning(f"Attempt {attempt + 1}: Non-JSON response format, retrying...")
            else:
                logging.warning(f"Attempt {attempt + 1}: Empty response, retrying...")
                
            time.sleep(2 ** min(attempt, 3))  # Exponential backoff, capped at 8 seconds
            
        except Exception as e:
            logging.warning(f"Azure GPT evaluation attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** min(attempt, 3))  # Exponential backoff

    logging.error(f"Azure GPT evaluation failed after {max_retries} attempts.")
    return ""

def extract_json_field(response, score_key, reason_key):
    """Extract score and reasoning from JSON response"""
    try:
        # Handle double braces from prompt format
        if response.startswith('{{') and response.endswith('}}'):
            response = response[1:-1]  # Remove outer braces
        
        # Try parsing as JSON first
        data = json.loads(response)
        score = data.get(score_key)
        reason = data.get(reason_key) or data.get("reasoning")
        return int(score) if score is not None else None, reason
    except:
        return None, None

def extract_score_and_reason(response, score_key, reason_fields, prefix_patterns=None):
    """Extract score and reasoning from GPT response with fallback patterns"""
    # Try JSON extraction first
    for reason_field in reason_fields:
        score, reason = extract_json_field(response, score_key, reason_field)
        if score is not None:
            return score, reason
    
    # Fallback to regex patterns
    patterns = [
        rf"{score_key}\s*[:：]?\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
    ]
    if prefix_patterns:
        patterns = prefix_patterns + patterns
    
    score = None
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    
    # Extract reasoning
    reason = None
    reason_match = re.search(r"reasoning\s*[:：]\s*(.+)", response, re.IGNORECASE|re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    
    return score, reason

def evaluate_images(image_id, metrics=None, vortex_data=None, api_key=None):
    """
    Evaluate causal reasoning images based on specified metrics
    """
    metrics = metrics or METRICS
    results = {}
    
    # Note: Azure API key is configured globally, api_key parameter ignored
    # Azure OpenAI client is already initialized with credentials
    
    # Find the specific task
    task = None
    for t in vortex_data["tasks"]:
        if t["id"] == image_id:
            task = t
            break
    
    if not task:
        logging.warning(f"Task ID {image_id} not found")
        return results

    # Get images from HF dataset (PIL Image objects) and file paths
    original_image = task.get('image')  # PIL Image object from HF dataset
    target_image = task.get('target_image')  # PIL Image object from HF dataset
    
    generated_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.png")
    think_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.txt")

    # Check if original image exists
    if original_image is None:
        logging.error(f"Original image not found for task {image_id}")
        return results
    if not os.path.isfile(generated_path):
        logging.error(f"Generated image not found: {generated_path}")
        return results
    
    # Load think output if exists
    think_output = None
    if os.path.isfile(think_path):
        try:
            with open(think_path, 'r', encoding='utf-8') as f:
                think_output = f.read().strip()
        except Exception as e:
            logging.warning(f"Error loading think output {think_path}: {e}")

    # Encode images
    orig_b64 = encode_image_to_base64(original_image)  # PIL Image object
    gen_b64 = encode_image_to_base64(generated_path)   # File path
    if not orig_b64 or not gen_b64:
        logging.error(f"Failed to encode images for {image_id}")
        return results
    
    # Encode target image if available
    target_b64 = None
    if target_image is not None:
        target_b64 = encode_image_to_base64(target_image)
    


    # Extract task information
    prompt = task.get("prompt", "")
    dimension = task.get("dimension", "")
    keywords = task.get("keywords", "")  # Already a string in HF dataset
    target_description = task.get("target_description", "")
    
    # Keywords is already a string, use directly
    keywords_str = keywords if keywords else ""

    # Evaluate each metric
    for metric in metrics:
        if metric == "reasoning_process":
            prompt_text = prompt_reasoning_process_causal.format(
                prompt=prompt,
                dimension=dimension,
                keywords=keywords_str,
                target_description=target_description,
                think_output=think_output if think_output else "No think output available"
            )
            # Retry logic for null scores
            score, reason = None, None
            for retry_attempt in range(3):  # Max 3 retries for null scores
                resp = evaluate_with_gpt(prompt_text, orig_b64, None)
                score, reason = extract_score_and_reason(resp, "reasoning_process_score", ["reasoning"])
                if score is not None:  # Success, break out
                    break
                logging.warning(f"Reasoning process evaluation attempt {retry_attempt + 1} returned null, retrying...")
            
            results["reasoning_process_score"] = score
            results["reasoning_process_reasoning"] = reason
            
        elif metric == "reasoning_visual":
            prompt_text = prompt_reasoning_visual_causal.format(
                prompt=prompt,
                dimension=dimension,
                keywords=keywords_str,
                target_description=target_description
            )
            resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64, target_b64)
            score, reason = extract_score_and_reason(resp, "reasoning_visual_score", ["reasoning"])
            results["reasoning_visual_score"] = score
            results["reasoning_visual_reasoning"] = reason

        elif metric == "reasoning_alignment":
            prompt_text = prompt_reasoning_alignment.format(
                prompt=prompt,
                think_output=think_output if think_output else "No think output available"
            )
            # Retry logic for null scores
            score, reason = None, None
            for retry_attempt in range(3):  # Max 3 retries for null scores
                resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64)
                score, reason = extract_score_and_reason(resp, "reasoning_alignment_score", ["reasoning"])
                if score is not None:  # Success, break out
                    break
                logging.warning(f"Reasoning alignment evaluation attempt {retry_attempt + 1} returned null, retrying...")
            
            results["reasoning_alignment_score"] = score
            results["reasoning_alignment_reasoning"] = reason

        elif metric == "visual_consistency":
            prompt_text = prompt_visual_consistency.format(prompt=prompt)
            resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64)
            score, reason = extract_score_and_reason(resp, "visual_consistency_score", ["reasoning"])
            results["visual_consistency_score"] = score
            results["visual_consistency_reasoning"] = reason

        elif metric == "image_quality":
            resp = evaluate_with_gpt(prompt_image_quality, None, gen_b64)
            score, reason = extract_score_and_reason(resp, "image_quality_score", ["reasoning"])
            results["image_quality_score"] = score
            results["image_quality_reasoning"] = reason

    return results

