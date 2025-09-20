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
    prompt_reasoning_process_logical,
    prompt_reasoning_visual_logical,
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

# VortexBench data paths
VORTEX_GEN_DIR = "/code/gen_banana"

def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file with thread lock"""
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {"key": key, "result": result}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys_with_missing_metrics(jsonl_path, metrics, expected_keys_map):
    """Load processed keys and identify missing metrics for each key"""
    processed_keys = set()
    missing_metrics_map = {}
    
    if not os.path.exists(jsonl_path):
        return processed_keys, missing_metrics_map
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = data.get('key')
                    result = data.get('result', {})
                    
                    if key:
                        processed_keys.add(key)
                        
                        # Check which metrics are missing for this key
                        missing_metrics = []
                        for metric in metrics:
                            if metric not in result:
                                missing_metrics.append(metric)
                        
                        if missing_metrics:
                            missing_metrics_map[key] = missing_metrics
                            
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.error(f"Error loading processed keys: {e}")
    
    return processed_keys, missing_metrics_map

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
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
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract score using regex
            score_match = re.search(r'(\d+(?:\.\d+)?)', content)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                if 0 <= score <= 100:
                    return score, content
                else:
                    logging.warning(f"Score {score} out of range, retrying...")
            else:
                logging.warning(f"No score found in response: {content[:100]}...")
                
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
    return 0, "Failed to get valid response after all retries"

def evaluate_reasoning_process(think_output, prompt, dimension, keywords, target_description):
    """Evaluate reasoning process quality"""
    if not think_output or think_output.strip() == "No think output available":
        return 0, "No reasoning text provided"
    
    prompt_text = prompt_reasoning_process_logical.format(
        prompt=prompt,
        dimension=dimension,
        keywords=keywords,
        target_description=target_description,
        think_output=think_output
    )
    
    score, explanation = evaluate_with_gpt(prompt_text)
    return score, explanation

def evaluate_reasoning_visual(original_image, generated_image, target_image, prompt, dimension, keywords, target_description):
    """Evaluate visual reasoning result"""
    # Encode images to base64
    orig_b64 = encode_image_to_base64(original_image)
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not orig_b64 or not gen_b64:
        return 0, "Failed to encode images"
    
    # Encode target image if available
    target_b64 = None
    if target_image is not None:
        target_b64 = encode_image_to_base64(target_image)
    
    prompt_text = prompt_reasoning_visual_logical.format(
        prompt=prompt,
        dimension=dimension,
        keywords=keywords,
        target_description=target_description
    )
    
    resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64, target_b64)
    return resp

def evaluate_reasoning_alignment(original_image, generated_image, think_output, prompt):
    """Evaluate alignment between reasoning text and visual result"""
    if not think_output or think_output.strip() == "No think output available":
        return 0, "No reasoning text provided"
    
    # Encode images to base64
    orig_b64 = encode_image_to_base64(original_image)
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not orig_b64 or not gen_b64:
        return 0, "Failed to encode images"
    
    prompt_text = prompt_reasoning_alignment.format(
        prompt=prompt,
        think_output=think_output
    )
    
    resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64)
    return resp

def evaluate_visual_consistency(original_image, generated_image, prompt):
    """Evaluate visual consistency"""
    # Encode images to base64
    orig_b64 = encode_image_to_base64(original_image)
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not orig_b64 or not gen_b64:
        return 0, "Failed to encode images"
    
    prompt_text = prompt_visual_consistency.format(prompt=prompt)
    
    resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64)
    return resp

def evaluate_image_quality(generated_image):
    """Evaluate image quality"""
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not gen_b64:
        return 0, "Failed to encode image"
    
    prompt_text = prompt_image_quality
    
    resp = evaluate_with_gpt(prompt_text, None, gen_b64)
    return resp

def evaluate_images(image_id, metrics, vortex_data, api_key=None):
    """Evaluate images for logical reasoning tasks"""
    try:
        # Get task data
        task_data = None
        for task in vortex_data["tasks"]:
            if task["id"] == image_id:
                task_data = task
                break
        
        if not task_data:
            logging.error(f"Task data not found for {image_id}")
            return {}
        
        # Get file paths
        generated_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.png")
        think_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.txt")
        
        # Check if generated image exists
        if not os.path.exists(generated_path):
            logging.error(f"Generated image not found: {generated_path}")
            return {}
        
        # Load think output if available
        think_output = "No think output available"
        if os.path.exists(think_path):
            try:
                with open(think_path, 'r', encoding='utf-8') as f:
                    think_output = f.read().strip()
            except Exception as e:
                logging.error(f"Error reading think output: {e}")
        
        # Get original image and target image
        original_image = task_data.get("image")
        target_image = task_data.get("target_image")
        
        if not original_image:
            logging.error(f"Original image not found for {image_id}")
            return {}
        
        # Save original image temporarily for evaluation
        original_path = f"/tmp/original_{image_id}.png"
        original_image.save(original_path)
        
        # Save target image temporarily if available
        target_path = None
        if target_image is not None:
            target_path = f"/tmp/target_{image_id}.png"
            target_image.save(target_path)
        
        result = {}
        
        # Evaluate each metric
        for metric in metrics:
            try:
                if metric == "reasoning_process":
                    score, explanation = evaluate_reasoning_process(
                        think_output, 
                        task_data["prompt"], 
                        task_data["dimension"], 
                        task_data["keywords"], 
                        task_data["target_description"]
                    )
                elif metric == "reasoning_visual":
                    score, explanation = evaluate_reasoning_visual(
                        original_path, 
                        generated_path, 
                        target_path, 
                        task_data["prompt"], 
                        task_data["dimension"], 
                        task_data["keywords"], 
                        task_data["target_description"]
                    )
                elif metric == "reasoning_alignment":
                    score, explanation = evaluate_reasoning_alignment(
                        original_path, 
                        generated_path, 
                        think_output, 
                        task_data["prompt"]
                    )
                elif metric == "visual_consistency":
                    score, explanation = evaluate_visual_consistency(
                        original_path, 
                        generated_path, 
                        task_data["prompt"]
                    )
                elif metric == "image_quality":
                    score, explanation = evaluate_image_quality(generated_path)
                else:
                    logging.warning(f"Unknown metric: {metric}")
                    continue
                
                result[metric] = {
                    "score": score,
                    "explanation": explanation
                }
                
            except Exception as e:
                logging.error(f"Error evaluating {metric} for {image_id}: {e}")
                result[metric] = {
                    "score": 0,
                    "explanation": f"Error: {str(e)}"
                }
        
        # Clean up temporary files
        try:
            os.remove(original_path)
            if target_path and os.path.exists(target_path):
                os.remove(target_path)
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {e}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in evaluate_images for {image_id}: {e}")
        return {}
