import llm_annotator.utils as utils
import pandas as pd

from typing import List


@utils.component("pre-process")
def pre_process_transcript(transcript_df: pd.DataFrame, obs_list: List[str] | str):
    if isinstance(obs_list, str) and obs_list == "all":
        obs_list = transcript_df["obsid"].unique().tolist()
    elif not isinstance(obs_list, list):
        raise ValueError(f"obs_list should be a list, got {type(obs_list)} instead.")

    transcript_df["obsid"] = transcript_df["obsid"].astype(str)
    transcript_df = transcript_df[transcript_df["obsid"].isin(obs_list)]
    return "transcript_df", transcript_df
