from llm_annotator.llm import batch_anthropic_annotate, batch_local_llm_annotate

from enum import Enum
from typing import Optional
from dataclasses import dataclass

class ModelType(Enum):
    GPT4O = "gpt-4o"
    O1 = "o1-preview"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE = "claude-3-5-sonnet-20241022"
    GEMINI = "gemini-1.5-pro"
    MISTRAL = "mistral--large-latest"
    DEEPSEEK = "deepseek-chat"
    LLAMA3B = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA70B = "meta-llama/Llama-3.3-70B-Instruct"
    LLAMA7B_LOCAL = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA13B_LOCAL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class ModelConfig:
    type: ModelType
    temperature: float = 0
    max_tokens: Optional[int] = None


model_configs = {
        "gpt-4o": ModelConfig(ModelType.GPT4O),
        "gpt-o1": ModelConfig(ModelType.O1),
        "claude-3-5": ModelConfig(ModelType.CLAUDE),
        "gemini-1.5-pro": ModelConfig(ModelType.GEMINI),
        "mistral": ModelConfig(ModelType.MISTRAL),
        "deepseek-v3": ModelConfig(ModelType.DEEPSEEK),
        "llama-3.2-3b": ModelConfig(ModelType.LLAMA3B),
        "llama-3.3-70b": ModelConfig(ModelType.LLAMA70B),
        "llama-3b-local": ModelConfig(ModelType.LLAMA7B_LOCAL, max_tokens=512),
        "llama-70b-local": ModelConfig(ModelType.LLAMA13B_LOCAL, max_tokens=512)
}

annotation_configs = {
    "claude-3-5": batch_anthropic_annotate,
    "llama-3b-local": lambda requests: batch_local_llm_annotate(requests, "meta-llama/Llama-3.2-3B-Instruct"),
    "llama-70b-local": lambda requests: batch_local_llm_annotate(requests, "meta-llama/Llama-3.3-70B-Instruct")
}