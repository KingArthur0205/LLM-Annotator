import inspect
import llm_annotator.registry as registry

from typing import Any, Callable


def valid_kwargs(kwargs: dict[str, Any], func: Callable) -> dict[str, Any]:
    args = list(inspect.signature(func).parameters)
    return {k: kwargs[k] for k in kwargs if k in args}
