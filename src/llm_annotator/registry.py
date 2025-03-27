from typing import List, Dict
import pandas as pd

from llm_annotator.pipeline import Pipeline
from llm_annotator.dataloader import DataLoader
from llm_annotator.utils import load_meta_file


def simple_llm_pipe(model_list: List[str],
                    obs_list: List[str],
                    transcript_path: str,
                    sheet_source: str,
                    feature: str,
                    if_wait: bool,
                    n_uttr: int = 1,
                    annotation_prompt_path: str = ""):
    dataloader = DataLoader(sheet_source=sheet_source,
                            transcript_path=transcript_path)
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline(config=config)
    pipe.add_pipe(name="load_transcript", idx=0)
    pipe.add_pipe(name="load_feature", idx=1)
    pipe.add_pipe(name="generate_features", idx=2)
    pipe.add_pipe(name="pre-process", idx=3)
    pipe.add_pipe(name="build_examples", idx=4)
    pipe.add_pipe(name="build_annotation_prompt", idx=5)
    pipe.add_pipe(name="process_observations", idx=6)
    pipe.add_pipe(name="process_requests", idx=7)
    if if_wait:
        pipe.add_pipe(name="fetch_batch", idx=8)
        pipe.add_pipe(name="save_results", idx=9)

    return pipe


def fetch_pipe(batch_dir: str, feature: str):
    if_wait = False
    metadata = load_meta_file(batch_dir=batch_dir, feature=feature)
    transcript_path = metadata.get("transcript_path", "")
    sheet_source = metadata.get("sheet_source", "")
    annotation_prompt_path = metadata.get("annotation_prompt_path", "")
    feature = metadata.get("feature", "")
    transcript_df = pd.read_csv(transcript_path)

    config = {key: value for key, value in locals().items()}

    pipe = Pipeline(config=config)
    pipe.add_pipe(name="fetch_batch", idx=0)
    pipe.add_pipe(name="save_results", idx=1)
    return pipe
