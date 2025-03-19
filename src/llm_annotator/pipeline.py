from llm_annotator.config import ModelType, ModelConfig
from llm_annotator.registry import components
from llm_annotator.utils import valid_kwargs

from typing import Callable


model_configs = {
        "gpt-4o": ModelConfig(ModelType.GPT4O),
        "gpt-o1": ModelConfig(ModelType.O1),
        "claude-3-5": ModelConfig(ModelType.CLAUDE),
        "gemini-1.5-pro": ModelConfig(ModelType.GEMINI),
        "mistral": ModelConfig(ModelType.MISTRAL),
        "deepseek-v3": ModelConfig(ModelType.DEEPSEEK),
        "llama-3.2-3b": ModelConfig(ModelType.LLAMA3B),
        "llama-3.3-70b": ModelConfig(ModelType.LLAMA70B)
}


class Pipeline:
    def __init__(self):
        self.components: list[Callable] = []

    def __call__(self, *args, **kwargs):
        inputs = None
        outputs = []
        for name, component in self.components:
            output = component(inputs)
            inputs = output
        outputs.append(output)

        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def add_pipe(self, name: str):
        component_factory = components.get(name)
        component = component_factory(**valid_kwargs(self.config, component_factory))
        new_element = (name, component)
        self.components.insert(new_element)
