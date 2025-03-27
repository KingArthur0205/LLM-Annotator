# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1
import os

from argparse import ArgumentParser
from typing import List, Dict
from openai import OpenAI

import llm_annotator.prompt_parser
import llm_annotator.annotator
import llm_annotator.postprocess
from llm_annotator.registry import fetch_pipe
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
        feature: str,
        transcript_path: str,
        sheet_source: str,
        n_uttr: int,
        if_wait=False
):
    pipe = simple_llm_pipe(model_list=model_list,
                           obs_list=obs_list,
                           feature=feature,
                           transcript_path=transcript_path,
                           sheet_source=sheet_source,
                           if_wait=if_wait,
                           n_uttr=n_uttr)
    return pipe()


def fetch(batch_dir: str = None,
          feature: str = None):

    pipe = fetch_pipe(batch_dir=batch_dir,
                      feature=feature)
    return pipe()


def set_working_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    os.chdir(repo_root)
    print(f"Working directory set to: {os.getcwd()}")


def main():
    set_working_dir()


    #outputs = annotate(model_list=["claude-3-7"],
    #                   obs_list=["146"],
    #                   feature="Mathcompetent",
    #                   transcript_path="./data/alltranscripts_423_clean_segmented.csv",
    #                   sheet_source="./data/MOL Roles Features.xlsx",
    #                   if_wait=False,
    #                   n_uttr=10)


    fetch(feature="Mathcompetent")


    # a = outputs["batch_results"]
    # print(a['gpt-4o'])


if __name__ == "__main__":
    main()
