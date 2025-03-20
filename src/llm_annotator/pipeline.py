import confection
import inspect

from llm_annotator.config import ModelType, ModelConfig
from llm_annotator.utils import valid_kwargs, components

from typing import Callable


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
            #outputs.append(output)
            print(output)
            self.config[output_name] = output

        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def add_pipe(self, name: str, idx: int):
        component_factory = components.get(name)
        component = component_factory(**valid_kwargs(self.config, component_factory))

        new_element = (name, component)
        self.components.insert(idx, new_element)

