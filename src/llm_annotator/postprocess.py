import pandas as pd
import os
import json

import llm_annotator.utils as utils
from typing import Dict


def extract_json_code_block(response_text):
    start_tag = "```json"
    end_tag = "```"

    start_index = response_text.find(start_tag)
    if start_index == -1:
        return None  # or raise an error

    start_index += len(start_tag)
    end_index = response_text.find(end_tag, start_index)
    if end_index == -1:
        return None  # or raise an error

    json_block = response_text[start_index:end_index].strip()
    return json_block


@utils.component("save_results")
def save_results(batch_results: Dict, transcript_df: pd.DataFrame, feature: str, timestamp: str = None, save_dir: str = None):
    if timestamp is None:
        if not save_dir:
            if not os.path.exists(f"result/{feature}"):
                raise FileNotFoundError("The result folder doesn't exist.")
            batch_dir = f"result/{feature}"
        # Read from a specified result folder
        else:
            batch_dir = save_dir + f"/result/{feature}"
        timestamp = os.path.join(batch_dir, utils.find_latest_dir(batch_dir))
    else:
        # Create "results" directory if it does not exist
        batch_dir = utils.create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)
    transcript_df = transcript_df.copy()

    for model, batch_content in batch_results.items():
        if model == "gpt-4o":
            try:
                # Ensure batch_content is a string and split into lines (each line is a JSON object)
                batch_lines = batch_content.strip().split("\n")
                for line in batch_lines:
                    try:
                        response = json.loads(line)  # Convert JSON string to dictionary

                        # Extract custom_id and ensure it follows the format "request_X"
                        body = response.get("response", {}).get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            message_content = choices[0].get("message", {}).get("content", "{}")
                            log_probs = choices[0].get("logprobs", {}).get("content", "{}")
                            try:
                                parsed_content = json.loads(message_content)  # Convert string to dictionary
                            except json.JSONDecodeError:
                                print(
                                    f"Skipping response at line {line}: Invalid JSON content - {message_content}")
                                continue

                            result_log_probs = []
                            for prob in log_probs:
                                if prob['token'] in ("0", "1"):
                                    result_log_probs.append(round(prob['logprob'], 3))

                            index = 0
                            for utt_id, value in parsed_content.items():
                                matching_row = transcript_df['uttid'] == utt_id
                                transcript_df.loc[matching_row, model] = value
                                transcript_df.loc[matching_row, f"{model}_logprob"] = result_log_probs[index]
                                index += 1
                    except Exception as e:
                        print(f"Error processing response line: {line}. Error: {e}")
            except Exception as e:
                print(f"Error reading batch content for model {model}: {e}")
        elif model == "claude-3-7":
            for response in batch_content:
                try:
                    response_text = response.result.message.content[0].text
                    if "```json" in response_text:
                        response_text = extract_json_code_block(response_text)
                        response_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                    response = json.loads(response_text)

                    for utt_id, value in response.items():
                        matching_row = transcript_df['uttid'] == utt_id
                        transcript_df.loc[matching_row, model] = value
                except:
                    print(f"Error processing {response}.")
                    continue

    # Save the annotated dataframe
    if batch_results:
        transcript_df.to_csv(f"{batch_dir}/atn_df.csv", index=False)
        if save_dir is not None:
            transcript_df.to_csv(f"{save_dir}/atn_df.csv", index=False)
    return "annotated_df", transcript_df
