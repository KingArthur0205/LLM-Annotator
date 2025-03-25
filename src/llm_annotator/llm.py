import instructor
import os
import json

import openai
import anthropic

from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request

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


def anthropic_annotate(prompt: str):
    client = instructor.from_anthropic(Anthropic())
    completion = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "system",
             "content": "You are an expert at structured data annotation. You will be given unstructured student dialogue from math class discussions and should annotate the utternace with appropriate values. Return the output in JSON format."},
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


def store_batch(batches: Dict, feature: str):
    results_dir = "result"
    feature_dir = os.path.join(results_dir, feature)
    os.makedirs(feature_dir, exist_ok=True)

    # Create batch dir
    batch_meta_dir = os.path.join(feature_dir, "batch_meta")
    os.makedirs(batch_meta_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    batch_dir = os.path.join(batch_meta_dir, f"{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
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
        
        with open(file_path, "w") as f:
            json.dump(batch_metadata, f, indent=2)
    return "batch_dir", batch_meta_dir
