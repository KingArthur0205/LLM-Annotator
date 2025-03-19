import functools
import inspect
import catalogue

from typing import List

from llm_annotator.pipeline import Pipeline

components = catalogue.create("llm_annotator", "components")


def component(name):
    def component_decorator(func):
        @functools.wraps(func)
        def component_factory(*args, **kwargs):
            return functools.partial(func, *args, **kwargs)

        # Make the factory signature identical to func's except for the first argument.
        signature = inspect.signature(func)
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        component_factory.__signature__ = signature
        components.register(name)(component_factory)
        return func

    return component_decorator


def simple_llm_pipe(model_list: List[str]):
    pipe = Pipeline()
    pipe.add_pipe("build_prompt")