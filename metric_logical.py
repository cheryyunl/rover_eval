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
from config import OPENAI_API_KEY, OPENAI_MODEL, AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION

# Initialize OpenAI client (prefer OpenAI over Azure)
if OPENAI_API_KEY:
    # Use standard OpenAI API
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    model_name = OPENAI_MODEL
else:
    # Fallback to Azure OpenAI
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    model_name = AZURE_DEPLOYMENT_NAME

# Define metrics for all reasoning types
METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment", "visual_consistency", "image_quality"]

# VortexBench data paths
VORTEX_GEN_DIR = "/Users/cheryunl/Documents/eval/gen_banana"

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
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=3000,
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
    """Extract score and reasoning from GPT response with robust parsing"""
    if not response or not response.strip():
        return None, None
    
    # Clean up response
    response = response.strip()
    
    # Try multiple JSON parsing strategies
    score, reason = None, None
    
    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response)
        score = data.get(score_key)
        for reason_field in reason_fields:
            reason = data.get(reason_field)
            if reason:
                break
        if score is not None:
            return int(score), reason
    except:
        pass
    
    # Strategy 2: Handle double braces
    try:
        if response.startswith('{{') and response.endswith('}}'):
            clean_response = response[1:-1]
            data = json.loads(clean_response)
            score = data.get(score_key)
            for reason_field in reason_fields:
                reason = data.get(reason_field)
                if reason:
                    break
            if score is not None:
                return int(score), reason
    except:
        pass
    
    # Strategy 3: Extract JSON from text
    try:
        # Look for JSON-like patterns in the text
        json_pattern = r'\{[^{}]*"' + re.escape(score_key) + r'"[^{}]*\}'
        json_match = re.search(json_pattern, response, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            score = data.get(score_key)
            for reason_field in reason_fields:
                reason = data.get(reason_field)
                if reason:
                    break
            if score is not None:
                return int(score), reason
    except:
        pass
    
    # Strategy 4: Regex fallback patterns
    patterns = [
        rf"{score_key}\s*[:：]?\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
        r"score\s*[:：]?\s*([1-5])",
        r"rating\s*[:：]?\s*([1-5])",
    ]
    if prefix_patterns:
        patterns = prefix_patterns + patterns
    
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    
    # Extract reasoning with multiple patterns
    reason_patterns = [
        r"reasoning\s*[:：]\s*(.+)",
        r"explanation\s*[:：]\s*(.+)",
        r"analysis\s*[:：]\s*(.+)",
        r"evaluation\s*[:：]\s*(.+)",
    ]
    
    for pat in reason_patterns:
        reason_match = re.search(pat, response, re.IGNORECASE|re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            # Clean up reasoning text
            if reason.startswith('"') and reason.endswith('"'):
                reason = reason[1:-1]
            break
    
    return score, reason

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
    
    resp = evaluate_with_gpt(prompt_text)
    score, reason = extract_score_and_reason(resp, "reasoning_process_score", ["reasoning"])
    return score if score is not None else 0, reason if reason else "No reasoning provided"

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
    score, reason = extract_score_and_reason(resp, "reasoning_visual_score", ["reasoning"])
    return score if score is not None else 0, reason if reason else "No reasoning provided"

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
    score, reason = extract_score_and_reason(resp, "reasoning_alignment_score", ["reasoning"])
    return score if score is not None else 0, reason if reason else "No reasoning provided"

def evaluate_visual_consistency(original_image, generated_image, prompt):
    """Evaluate visual consistency"""
    # Encode images to base64
    orig_b64 = encode_image_to_base64(original_image)
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not orig_b64 or not gen_b64:
        return 0, "Failed to encode images"
    
    prompt_text = prompt_visual_consistency.format(prompt=prompt)
    
    resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64)
    score, reason = extract_score_and_reason(resp, "visual_consistency_score", ["reasoning"])
    return score if score is not None else 0, reason if reason else "No reasoning provided"

def evaluate_image_quality(generated_image):
    """Evaluate image quality"""
    gen_b64 = encode_image_to_base64(generated_image)
    
    if not gen_b64:
        return 0, "Failed to encode image"
    
    prompt_text = prompt_image_quality
    
    resp = evaluate_with_gpt(prompt_text, None, gen_b64)
    score, reason = extract_score_and_reason(resp, "image_quality_score", ["reasoning"])
    return score if score is not None else 0, reason if reason else "No reasoning provided"

def evaluate_images(image_id, metrics, vortex_data, api_key=None):
    """
    Evaluate logical reasoning images based on specified metrics
    """
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
