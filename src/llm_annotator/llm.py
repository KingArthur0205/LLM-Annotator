import instructor
import os
import json
import re

import anthropic

from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional

from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request

from llm_annotator.utils import create_batch_dir

# Local LLM imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

_ = load_dotenv(find_dotenv())


class Annotation(BaseModel):
    feature_name: str
    annotation: int


def openai_annotate(prompt: str):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "You are an expert at structured data annotation. You will be given unstructured student dialogue from math class discussions and should annotate the utternace with appropriate values. Return the output in JSON format."},
            {"role": "user", "content": f"{prompt}"}
        ],
        response_format=Annotation.model_json_schema(),
    )

    annotations = completion.choices[0].message.parsed
    return annotations


def anthropic_annotate(prompt: str, system_prompt: str):
    client = instructor.from_anthropic(Anthropic())
    completion = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=300,
        response_model=Annotation.model_json_schema(),
    )

    return completion


def batch_openai_annotate(requests: List[Dict]):
    os.makedirs("temp", exist_ok=True)  # Ensure the directory exists
    file_path = "temp/batch_input.jsonl"

    with open(file_path, "w") as f:
        for item in requests:
            json.dump(item, f)  # Convert dict to JSON string
            f.write("\n")

    client = OpenAI()

    # Upload the batch file
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_file = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Annotation job."
        }
    )
    return batch_file


def batch_anthropic_annotate(requests: List[Request]):
    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=requests
    )
    return message_batch


# Global cache for local models to avoid reloading
_local_model_cache = {}


def _get_local_model(model_name: str):
    """Get or load a local model with caching."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models. Install with: pip install transformers torch accelerate")
    
    if model_name not in _local_model_cache:
        print(f"Loading local model: {model_name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        _local_model_cache[model_name] = {
            "tokenizer": tokenizer,
            "model": model
        }
        print(f"Model {model_name} loaded successfully")
    
    return _local_model_cache[model_name]


def _format_llama_prompt(prompt: str, system_prompt: str = None) -> str:
    """Format prompt for Llama models using the chat template."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    return messages


def _extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from model response."""
    # Try to find JSON in the response
    json_pattern = r'\{[^{}]*"feature_name"[^{}]*"annotation"[^{}]*\}'
    match = re.search(json_pattern, response)
    
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract values manually
    feature_match = re.search(r'"feature_name":\s*"([^"]*)"', response)
    annotation_match = re.search(r'"annotation":\s*(\d+)', response)
    
    if feature_match and annotation_match:
        return {
            "feature_name": feature_match.group(1),
            "annotation": int(annotation_match.group(1))
        }
    
    return None


def local_llm_annotate(prompt: str, model_name: str, system_prompt: str = None) -> Annotation:
    """Annotate using a local LLM model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models")
    
    # Get the cached model
    model_cache = _get_local_model(model_name)
    tokenizer = model_cache["tokenizer"]
    model = model_cache["model"]
    
    # Format the prompt
    messages = _format_llama_prompt(prompt, system_prompt)
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to device if using GPU
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Extract JSON from response
    json_data = _extract_json_from_response(response)
    
    if json_data:
        return Annotation(**json_data)
    else:
        # Fallback annotation if parsing fails
        return Annotation(feature_name="unknown", annotation=0)


def batch_local_llm_annotate(requests: List[Dict], model_name: str) -> List[Dict]:
    """Batch annotate using a local LLM model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for local models")
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            # Extract prompt and system prompt from request
            messages = request.get("body", {}).get("messages", [])
            system_prompt = None
            user_prompt = None
            
            for message in messages:
                if message.get("role") == "system":
                    system_prompt = message.get("content", "")
                elif message.get("role") == "user":
                    user_prompt = message.get("content", "")
            
            if user_prompt:
                annotation = local_llm_annotate(user_prompt, model_name, system_prompt)
                result = {
                    "custom_id": request.get("custom_id", f"request_{i}"),
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": json.dumps({
                                        "feature_name": annotation.feature_name,
                                        "annotation": annotation.annotation
                                    })
                                }
                            }]
                        }
                    }
                }
                results.append(result)
            
        except Exception as e:
            print(f"Error processing request {i}: {str(e)}")
            # Add error result
            result = {
                "custom_id": request.get("custom_id", f"request_{i}"),
                "response": {
                    "body": {
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "feature_name": "error",
                                    "annotation": 0
                                })
                            }
                        }]
                    }
                }
            }
            results.append(result)
    
    return results


def store_meta(model_list: List[str],
               feature: str,
               obs_list: List[str],
               transcript_source: str,
               sheet_source: str,
               if_wait: bool,
               n_uttr: int,
               annotation_prompt_path: str,
               timestamp: str,
               prompt_template: str,
               system_prompt: str,
               save_dir: str):
    timestamp_dir = create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)

    # Create metadata dictionary
    metadata = {
        "model_list": model_list,
        "feature": feature,
        "obs_list": obs_list,
        "transcript_source": transcript_source,
        "sheet_source": sheet_source,
        "save_dir": timestamp_dir,
        "if_wait": if_wait,
        "n_uttr": n_uttr,
        "annotation_prompt_path": annotation_prompt_path,
        "timestamp": timestamp,
        "prompt": f"{prompt_template}",
        "system prompt": system_prompt
    }

    # Save metadata as JSON
    meta_file_path = os.path.join(timestamp_dir, "metadata.json")
    with open(meta_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {meta_file_path}")


def store_batch(batches: Dict,
                feature: str,
                save_dir: str,
                timestamp: str):
    batch_dir = create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)
    for model, batch_file in batches.items():
        batch_filename = f"{model}.json"
        file_path = os.path.join(batch_dir, batch_filename)

        if model == "gpt-4o":
            batch_metadata = {
                "id": batch_file.id,
                "object": batch_file.object,
                "endpoint": batch_file.endpoint,
                "errors": batch_file.errors,
                "input_file_id": batch_file.input_file_id,
                "completion_window": batch_file.completion_window,
                "status": batch_file.status,
                "output_file_id": batch_file.output_file_id,
                "error_file_id": batch_file.error_file_id,
                "created_at": batch_file.created_at,
                "in_progress_at": batch_file.in_progress_at,
                "expires_at": batch_file.expires_at,
                "completed_at": batch_file.completed_at,
                "failed_at": batch_file.failed_at,
                "expired_at": batch_file.expired_at,
                "stored_at": datetime.now().isoformat()
            }
        elif model == "claude-3-7":
            # Convert MessageBatchRequestCounts to a dictionary
            batch_metadata = {
                "id": batch_file.id,
                "type": batch_file.type,
                "processing_status": batch_file.processing_status,
                "request_counts": {
                    "processing": batch_file.request_counts.processing,
                    "succeeded": batch_file.request_counts.succeeded,
                    "errored": batch_file.request_counts.errored,
                    "canceled": batch_file.request_counts.canceled,
                    "expired": batch_file.request_counts.expired
                },
                "ended_at": batch_file.ended_at,
                "created_at": batch_file.created_at.isoformat(),
                "expires_at": batch_file.expires_at.isoformat(),
                "cancel_initiated_at": batch_file.cancel_initiated_at,
                "results_url": batch_file.results_url,
                "stored_at": datetime.now().isoformat()
            }
        elif model in ["llama-3b-local", "llama-70b-local"]:
            # For local models, batch_file is a list of results
            batch_metadata = {
                "model": model,
                "type": "local_batch",
                "processing_status": "completed",
                "request_counts": {
                    "processing": 0,
                    "succeeded": len(batch_file),
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0
                },
                "total_requests": len(batch_file),
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "stored_at": datetime.now().isoformat()
            }
        
        with open(file_path, "w") as f:
            json.dump(batch_metadata, f, indent=2)
    return "batch_dir", batch_dir
