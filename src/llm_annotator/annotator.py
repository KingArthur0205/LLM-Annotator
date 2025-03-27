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
from llm_annotator.llm import Annotation, batch_anthropic_annotate, batch_openai_annotate, store_batch, store_meta
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
                                     "a list of student dialogue utterances from math class discussions and should "
                                     "annotate the utterance and return the full list. You must provide results in JSON format."}],
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
                                                      "a list of student dialogue utterances from math class discussions and should "
                                                      "annotate the utterance and return the full list. You must provide results in JSON format."},
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

    # Group data by observation ID for faster context window construction
    obs_groups = group_obs(if_context=if_context, obs_list=obs_list, transcript_df=transcript_df)

    # Process eligible rows
    annotated_set = set()
    model_reqs = {}
    for model in model_list:
        model_reqs[model] = []

    for i in range(0, len(eligible_rows), n_uttr):
        # Build context window if requested
        # window = ""
        # if if_context:
        #    window = self._build_context_window(row, atn_df, obs_groups, fwd_window, bwd_window)
        batch_uttr = eligible_rows.iloc[i:i + n_uttr]

        combined_dialogue = "\n".join(
            f"{row['uttid']}: {row['dialogue']}"
            for j, row in batch_uttr.iterrows()
        )

        prompt = prompt_template.format(dialogue=combined_dialogue)
        # Get annotations from each model
        for model in model_list:
            # Create the model
            request = create_request(model=model, prompt=prompt, idx=i)
            model_reqs[model].append(request)

    return "model_requests", model_reqs


@utils.component("process_requests")
def process_requests(model_requests: Dict,
                     feature: str,
                     model_list: List[str],
                     obs_list: List[str],
                     transcript_path: str,
                     sheet_source: str,
                     if_wait: bool,
                     n_uttr: int,
                     annotation_prompt_path: str
                     ) -> Dict:
    batches = {}
    for model, req_list in model_requests.items():
        req_list = req_list[:100]

        if model == "gpt-4o":
            batch = batch_openai_annotate(requests=req_list)

        elif model == "claude-3-7":
            batch = batch_anthropic_annotate(requests=req_list)
            
        batches[model] = batch
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    store_batch(batches=batches, feature=feature, timestamp=timestamp)
    store_meta(feature=feature, model_list=model_list, obs_list=obs_list, transcript_path=transcript_path,
               sheet_source=sheet_source, if_wait=if_wait, n_uttr=n_uttr, annotation_prompt_path=annotation_prompt_path,
               timestamp=timestamp)

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
                elif status == "failed":
                    print(f"{model}: Batch {batch_id} has failed.")
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
                elif not status == "ended":
                    print(f"{model}: Batch {batch_id} is still in progress.")

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
