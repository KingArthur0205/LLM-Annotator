# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1
import os

from argparse import ArgumentParser
from typing import List, Dict
from openai import OpenAI

import llm_annotator.prompt_parser
import llm_annotator.annotator
from llm_annotator.pipeline import Pipeline
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from llm_annotator.dataloader import DataLoader
from llm_annotator.registry import simple_llm_pipe
from llm_annotator import utils, preprocess
from llm_annotator.llm import openai_annotate, anthropic_annotate, batch_anthropic_annotate


def annotate(
        model_list: List[str],
        obs_list: List[str],
        feature_list: List[str],
        transcript_path: str,
        sheet_source: str
):
    dataloader = DataLoader(sheet_source=sheet_source,
                            transcript_path=transcript_path)

    # Read in the feature file and transcript file
    feature_df = dataloader.get_features()
    transcript_df = dataloader.get_transcript()
    feature_dict = dataloader.generate_features(feature_list)

    pipe = simple_llm_pipe(model_list=model_list,
                           obs_list=obs_list,
                           feature_dict=feature_dict,
                           feature=feature_list[0],
                           transcript_df=transcript_df,
                           feature_df=feature_df)
    return pipe()


def set_working_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    os.chdir(repo_root)
    print(f"Working directory set to: {os.getcwd()}")


def main():
    set_working_dir()
    annotate(model_list=["gpt-4o"],
             obs_list=["146"],
             feature_list=["Mathcompetent"],
             transcript_path="./data/alltranscripts_423_clean_segmented.csv",
             sheet_source="./data/MOL Roles Features.xlsx")


if __name__ == "__main__":
    main()
