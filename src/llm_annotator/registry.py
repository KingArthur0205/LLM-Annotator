from typing import List, Dict

import pandas as pd
from llm_annotator.pipeline import Pipeline


def simple_llm_pipe(model_list: List[str],
                    feature_dict: Dict,
                    feature: str):
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline(config=config)
    pipe.add_pipe(name="build_example", idx=0)
    return pipe
