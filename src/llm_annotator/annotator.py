import pandas as pd
import time
import os
import json

from typing import Dict, List
from tqdm import tqdm

import anthropic
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_annotator import utils
from llm_annotator.llm import Annotation, batch_anthropic_annotate, batch_openai_annotate, store_batch
from llm_annotator.utils import find_latest_dir, Batch, load_batch_files


def mark_ineligible_rows(model_list: List[str],
                         feature_dict: Dict,
                         transcript_df: pd.DataFrame,
                         min_len):
    # Create separate dfs for individual features
    atn_feature_dfs = {feature_name: transcript_df.copy() for feature_name in feature_dict.keys()}

    # Filter out ineligible rows
    eligible_rows = transcript_df[(transcript_df['role'] == 'student') &
                                  (transcript_df['dialogue'].str.split().str.len() >= min_len)]
    ineligible_rows = transcript_df.index.difference(eligible_rows.index)

    # Mark ineligible rows with Nones
    for model_name in model_list:
        for feature_name in feature_dict:
            for idx in ineligible_rows:
                atn_feature_dfs[feature_name].at[idx, model_name] = None

    return eligible_rows, atn_feature_dfs


def group_obs(transcript_df: pd.DataFrame,
              if_context: bool,
              obs_list: List[str]):
    obs_groups = {}
    if if_context:
        for obs_id in obs_list:
            obs_groups[obs_id] = transcript_df[transcript_df['obsid'] == obs_id].index.tolist()
    return obs_groups


def create_request(model: str, prompt: str, idx: int):
    match model:
        case "claude-3-7":
            return Request(
                custom_id=f"request_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    system=[{"type": "text",
                             "text": "You are an expert at structured data annotation. You will be given "
                                     "unstructured student dialogue from math class discussions and should "
                                     "annotate the utterance. You must provide results in JSON format.\nExample: {'Mathcompetent': 0}"}],
                    messages=[{"role": "user",
                               "content": prompt,
                               }]
                )
            )
        case "gpt-4o":
            return {"custom_id": f"request_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-4o",
                             "messages": [{"role": "system",
                                           "content": "You are an expert at structured data annotation. You will be given "
                                           "unstructured student dialogue from math class discussions and should "
                                           "annotate the utterance. You must provide results in JSON format.\nExample: {'Mathcompetent': 0}"},
                                          {"role": "user",
                                           "content": prompt}],
                             "max_tokens": 1000,
                             "temperature": 0,
                             "response_format": {
                                    "type": "json_object"
                             }}
                    }


@utils.component("process_observations")
def process_observations(transcript_df: pd.DataFrame,
                         model_list: List[str],
                         feature_dict: Dict,
                         prompt_template: str,
                         obs_list: List[str] = None,
                         if_context: bool = False,
                         fwd_window: int = 0,
                         bwd_window: int = 0,
                         min_len: int = 6,
                         **kwargs) -> Dict[str, pd.DataFrame]:

    # Create a dictionary to store results per feature
    eligible_rows, atn_feature_dfs = mark_ineligible_rows(model_list=model_list,
                                                          feature_dict=feature_dict,
                                                          transcript_df=transcript_df,
                                                          min_len=min_len)

    # Group data by observation ID for faster context window construction
    obs_groups = group_obs(if_context=if_context, obs_list=obs_list, transcript_df=transcript_df)

    # Process eligible rows
    annotated_set = set()
    model_reqs = {}
    for model in model_list:
        model_reqs[model] = []

    for idx, row in eligible_rows.iterrows():
        # Build context window if requested
        # window = ""
        # if if_context:
        #    window = self._build_context_window(row, atn_df, obs_groups, fwd_window, bwd_window)

        prompt = prompt_template.format(dialogue=row['dialogue'])
        # Get annotations from each model
        for model in model_list:
            # Create the model
            request = create_request(model=model, prompt=prompt, idx=idx)
            model_reqs[model].append(request)

    return "model_requests", model_reqs


@utils.component("process_requests")
def process_requests(model_requests: Dict, feature: str) -> Dict:
    batches = {}
    for model, req_list in model_requests.items():
        req_list = req_list[:2]

        if model == "gpt-4o":
            batch = batch_openai_annotate(requests=req_list)

        elif model == "claude-3-7":
            batch = batch_anthropic_annotate(requests=req_list)
            
        batches[model] = batch
    store_batch(batches=batches, feature=feature)
    return "batches", batches


@utils.component("fetch_batch")
def fetch_batch(batches: Dict = None,
                batch_dir: str = None,
                feature: str = "",
                if_wait: bool = True):
    results = {}
    print("Fetching results...")

    if not batches:
        batches = load_batch_files(batch_dir, feature)

    # Define the function that processes batches and updates results
    def process_batches():
        all_done = True

        for model, batch in batches.items():
            batch_id = batch.id
            if model == "gpt-4o":
                client = openai.OpenAI()
                response = client.batches.retrieve(batch_id)
                status = response.status

                # Retrieve completed results
                if status == "completed":
                    result = client.files.content(response.output_file_id).read().decode("utf-8")
                    print(f"Results for {model} is completed")
                    results[model] = result

            elif model == "claude-3-7":
                client = anthropic.Anthropic()
                response = client.messages.batches.retrieve(batch_id)
                status = response.processing_status
                results[model] = []

                if status == "ended":
                    print(f"Results for {model} is completed")
                    # Initialize entry variable

                    for result in client.messages.batches.results(message_batch_id=batch_id):
                        if result.result.type == "succeeded":
                            # Access message content properly - assuming text is inside a content list
                            results[model].append(result)
                        elif result.result.type == "error":
                            print(f"Error for {result.custom_id}: {result.result.error}")
                        else:
                            print(f"Unexpected result type: {result.result.type}")

            if status not in ["completed", "failed", "cancelled"]:
                all_done = False  # Keep checking if not finished

        return all_done

    if if_wait:
        # Use the loop to keep checking until all batches are done
        while True:
            all_done = process_batches()

            if all_done:
                print("All annotation tasks are finished.")
                break  # Exit loop if all batches are done

            time.sleep(10)
    else:
        # Execute the batch processing just once without waiting
        process_batches()

    return "batch_results", results
