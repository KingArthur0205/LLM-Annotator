from pydantic import BaseModel
from openai import OpenAI

import os

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
