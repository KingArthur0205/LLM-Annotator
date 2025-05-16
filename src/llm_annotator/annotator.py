import pandas as pd
import time
import os
import json


from typing import Dict, List
from datetime import datetime

import anthropic
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_annotator import utils
from llm_annotator.llm import batch_anthropic_annotate, batch_openai_annotate, store_batch, store_meta
from llm_annotator.utils import load_batch_files


def mark_ineligible_rows(model_list: List[str],
                         feature_dict: Dict,
                         transcript_df: pd.DataFrame,
                         min_len):
    # Create separate dfs for individual features
    atn_feature_dfs = {feature_name: transcript_df.copy() for feature_name in feature_dict.keys()}

    # Filter out ineligible rows
    eligible_rows = transcript_df[(transcript_df['role'].str.lower() == 'student') &
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


def create_request(model: str, prompt: str, system_prompt: str, idx: int):
    match model:
        case "claude-3-7":
            return Request(
                custom_id=f"request_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    system=[{"type": "text",
                             "text": system_prompt}],
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
                                           "content": system_prompt},
                                          {"role": "user",
                                           "content": prompt}],
                             "max_tokens": 1000,
                             "logprobs": True,
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
                         system_prompt: str,
                         n_uttr: int,
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
    print("Hi")
    print(eligible_rows)

    # Group data by observation ID for faster context window construction
    obs_groups = group_obs(if_context=if_context, obs_list=obs_list, transcript_df=transcript_df)

    # Process eligible rows
    annotated_set = set()
    model_reqs = {}
    for model in model_list:
        model_reqs[model] = []

    i = 0
    while i < len(eligible_rows):
        # Get current segment ID
        current_segment_id = eligible_rows.iloc[i]['segment_id_1sd']

        # Find the end index - either after n_uttr rows or when segment changes
        max_idx = min(i + n_uttr, len(eligible_rows))
        end_idx = i

        for j in range(i, max_idx):
            if eligible_rows.iloc[j]['segment_id_1sd'] != current_segment_id:
                break
            end_idx = j + 1

        # Extract the batch of utterances
        batch_uttr = eligible_rows.iloc[i:end_idx]

        # Create combined dialogue from batch
        combined_dialogue = "\n".join(
            f"{row['uttid']}: {row['dialogue']}"
            for _, row in batch_uttr.iterrows()
        )

        # Create prompt and model requests
        prompt = prompt_template.format(dialogue=combined_dialogue)

        for model in model_list:
            request = create_request(model=model, prompt=prompt, system_prompt=system_prompt, idx=i)
            model_reqs[model].append(request)

        # Update index for next iteration
        i = end_idx if end_idx > i else i + n_uttr

    return "model_requests", model_reqs


@utils.component("process_requests")
def process_requests(model_requests: Dict,
                     feature: str,
                     model_list: List[str],
                     obs_list: List[str],
                     transcript_source: str,
                     sheet_source: str,
                     if_wait: bool,
                     n_uttr: int,
                     annotation_prompt_path: str,
                     prompt_template: str,
                     system_prompt: str,
                     timestamp: str,
                     save_dir: str,
                     if_test: bool = False
                     ) -> Dict:
    batches = {}
    for model, req_list in model_requests.items():
        req_list = req_list[:100] if if_test else req_list

        if model == "gpt-4o":
            batch = batch_openai_annotate(requests=req_list)

        elif model == "claude-3-7":
            batch = batch_anthropic_annotate(requests=req_list)
            
        batches[model] = batch

    store_batch(batches=batches, feature=feature, timestamp=timestamp, save_dir=save_dir)
    store_meta(feature=feature, model_list=model_list, obs_list=obs_list, transcript_source=transcript_source,
               sheet_source=sheet_source, if_wait=if_wait, n_uttr=n_uttr, annotation_prompt_path=annotation_prompt_path,
               timestamp=timestamp, prompt_template=prompt_template, system_prompt=system_prompt, save_dir=save_dir)

    return "batches", batches


@utils.component("fetch_batch")
def fetch_batch(save_dir: str,
                batches: Dict = None,
                timestamp: str = None,
                feature: str = "",
                if_wait: bool = True):
    results = {}
    print("Fetching results...")

    if not batches:
        batches = load_batch_files(timestamp=timestamp, feature=feature, save_dir=save_dir)
    if_gpt_finished = False if "gpt-4o" in batches.keys() else True
    if_claude_finished = False if "claude-3-7" in batches.keys() else True

    # Define the function that processes batches and updates results
    def process_batches(if_gpt_finished: bool, if_claude_finished: bool):
        for model, batch in batches.items():
            batch_id = batch.id
            if model == "gpt-4o":
                client = openai.OpenAI()
                response = client.batches.retrieve(batch_id)
                status = response.status

                # Retrieve completed results
                if status == "completed" and not if_gpt_finished:
                    result = client.files.content(response.output_file_id).read().decode("utf-8")
                    print(f"{model} has completed batching.")
                    results[model] = result
                    if_gpt_finished = True
                elif status == "expired":
                    print(f"{model}: Batch {batch_id} has expired.")
                    if_gpt_finished = True
                elif status == "failed":
                    print(f"{model}: Batch {batch_id} has failed.")
                    if_gpt_finished = True
                elif status == "in_progress":
                    print(f"{model}: Batch {batch_id} is still in progress.")

            elif model == "claude-3-7":
                client = anthropic.Anthropic()
                status = client.messages.batches.retrieve(batch_id).processing_status
                results[model] = []

                if status == "ended" and not if_claude_finished:
                    print(f"{model} has completed batching.")
                    if_claude_finished = True
                    # Initialize entry variable

                    for result in client.messages.batches.results(message_batch_id=batch_id):
                        if result.result.type == "succeeded":
                            # Access message content properly - assuming text is inside a content list
                            results[model].append(result)
                        elif result.result.type == "error":
                            print(f"Error for {result.custom_id}: {result.result.error}")
                        else:
                            print(f"Unexpected result type: {result.result.type}")
                elif status == "expired":
                    print(f"{model}: Batch {batch_id} has expired.")
                    if_claude_finished = True
                elif not status == "ended":
                    print(f"{model}: Batch {batch_id} is still in progress.")
                    if_claude_finished = True

        return if_gpt_finished, if_claude_finished

    if if_wait:
        # Use the loop to keep checking until all batches are done
        while True:
            if_gpt_finished, if_claude_finished = process_batches(if_gpt_finished, if_claude_finished)

            if if_gpt_finished and if_claude_finished:
                print("All annotation tasks are finished.")
                break  # Exit loop if all batches are done

            time.sleep(10)
    else:
        # Execute the batch processing just once without waiting
        process_batches(if_gpt_finished, if_claude_finished)

    return "batch_results", results
