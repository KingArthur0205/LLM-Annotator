import confection
import inspect

from llm_annotator.config import ModelType, ModelConfig
from llm_annotator.utils import valid_kwargs, components

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
    def __init__(self, config: dict[str, dict[str, str]]):
        self.components: list[Callable] = []
        self.config = confection.Config(config)

    def __call__(self, *args, **kwargs):
        outputs = []
        for name, component in self.components:
            component_params = inspect.signature(component).parameters
            component_kwargs = {param: self.config[param] for param in component_params if param in self.config}

            # Compute the outputs
            output_name, output = component(**component_kwargs)
            outputs.append(output)

            self.config[output_name] = output

        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def add_pipe(self, name: str, idx: int):
        component_factory = components.get(name)
        component = component_factory(**valid_kwargs(self.config, component_factory))

        new_element = (name, component)
        self.components.insert(idx, new_element)

