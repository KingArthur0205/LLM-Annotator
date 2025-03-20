import instructor
import os
import json

import openai
import anthropic

from typing import List, Dict
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


class Annotation(BaseModel):
    feature_name: str
    annotation: int


def openai_annotate(prompt: str):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "You are an expert at structured data annotation. You will be given unstructured student dialogue from math class discussions and should annotate the utternace with appropriate values."},
            {"role": "user", "content": f"{prompt}"}
        ],
        response_format=Annotation,
    )

    annotations = completion.choices[0].message.parsed
    return annotations


def anthropic_annotate(prompt: str):
    client = instructor.from_anthropic(Anthropic())
    completion = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "system",
             "content": "You are an expert at structured data annotation. You will be given unstructured student dialogue from math class discussions and should annotate the utternace with appropriate values."},
            {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens=300,
        response_model=Annotation,
    )

    return completion


def batch_openai_annotate(requests: List[Dict]):
    with open("temp/batch_input.jsonl", "w") as f:
        for item in requests:
            json.dump(item, f)  # Convert dict to JSON string
            f.write("\n")
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open("temp/batch_input.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    return client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Annotation job."
        }
    )


def batch_anthropic_annotate(requests: List[Request]):
    client = instructor.from_anthropic(Anthropic())

    message_batch = client.messages.batches.create(
        requests=requests
    )
    return message_batch
