import pandas as pd
import time

from typing import Dict, List
from tqdm import tqdm

import anthropic
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_annotator import utils
from llm_annotator.llm import Annotation, batch_anthropic_annotate, batch_openai_annotate


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
                    messages=[{"role": "system",
                               "content": "You are an expert at structured data annotation. You will be given "
                                          "unstructured student dialogue from math class discussions and should "
                                          "annotate the utterance provide results in JSON format."},
                              {"role": "user",
                               "content": prompt,
                               }],
                    response_model=Annotation.model_json_schema()
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
                                           "annotate the utterance and provide results in JSON format."},
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
def process_requests(model_requests: Dict) -> Dict:
    batches = {}
    for model, req_list in model_requests.items():
        req_list = req_list[:2]

        if model == "gpt-4o":
            batch = batch_openai_annotate(requests=req_list)

        elif model == "claude-3-7":
            batch = batch_anthropic_annotate(requests=req_list)
        batches[model] = batch
    return "batches", batches


@utils.component("fetch_batch")
def fetch_batch(batches: Dict):
    results = {}
    print("Fetching results...")

    for model, batch in batches.items():
        print(f"{model}: batch")

    while True:
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
                response = client.get_batch(batch_id)
                status = response.get("status")

                if status == "completed":
                    result = client.get_batch_results(batch_id)
                    print(f"Results for {model} is completed")
                    results[model] = result

            if status not in ["completed", "failed", "cancelled"]:
                all_done = False  # Keep checking if not finished

        if all_done:
            print("All annotation tasks are finished.")
            break  # Exit loop if all batches are done

        time.sleep(10)
    return "batch_results", results
