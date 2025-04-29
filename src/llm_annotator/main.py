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
        obs_list: List[str] | str,
        feature: str,
        transcript_path: str,
        sheet_source: str,
        n_uttr: int,
        if_wait=False,
        system_prompt_path: str = "data/prompts/system_prompt.txt",
        prompt_path: str = "",
        mode: str = ""
):
    pipe = simple_llm_pipe(model_list=model_list,
                           obs_list=obs_list,
                           feature=feature,
                           transcript_path=transcript_path,
                           system_prompt_path=system_prompt_path,
                           prompt_path=prompt_path,
                           sheet_source=sheet_source,
                           if_wait=if_wait,
                           n_uttr=n_uttr)
    pipe()


def fetch(batch_dir: str = None,
          feature: str = None,
          save_dir: str = None):

    pipe = fetch_pipe(batch_dir=batch_dir,
                      feature=feature)
    pipe()


def set_working_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    os.chdir(repo_root)
    print(f"Working directory set to: {os.getcwd()}")


def main():
    outputs = annotate(model_list=["gpt-4o"],
                       obs_list=["146"],
                       feature="Mathcompetent",
                       transcript_path="./data/alltranscripts_423_clean_segmented.csv",
                       sheet_source="./data/MOL Roles Features.xlsx",
                       prompt_path="data/prompts/base.txt",
                       if_wait=True,
                       n_uttr=10,
                       mode="CoT")


    #fetch(feature="Mathcompetent")


set_working_dir()

if __name__ == "__main__":
    main()
