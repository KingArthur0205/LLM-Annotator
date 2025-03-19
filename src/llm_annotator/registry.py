from typing import List, Dict

import pandas as pd
from llm_annotator.pipeline import Pipeline


def simple_llm_pipe(model_list: List[str],
                    obs_list: List[str],
                    feature_dict: Dict,
                    feature_df: Dict,
                    transcript_df: pd.DataFrame,
                    feature: str,
                    annotation_prompt_path: str = ""):
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline(config=config)
    pipe.add_pipe(name="pre-process", idx=0)
    pipe.add_pipe(name="build_examples", idx=1)
    pipe.add_pipe(name="build_annotation_prompt", idx=2)
    pipe.add_pipe(name="process_observations", idx=3)

    return pipe
