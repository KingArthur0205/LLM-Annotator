import pandas as pd
import os
import json

import llm_annotator.utils as utils
from typing import Dict


@utils.component("save_results")
def save_results(batch_results: Dict, transcript_df: pd.DataFrame, feature: str):
    # Create "results" directory if it does not exist
    results_dir = "result"
    feature_dir = os.path.join(results_dir, feature)
    os.makedirs(feature_dir, exist_ok=True)

    for model, batch_content in batch_results.items():
        if model == "gpt-4o":
            try:
                # Ensure batch_content is a string and split into lines (each line is a JSON object)
                batch_lines = batch_content.strip().split("\n")

                print(f"Processing model: {model}, Batch size: {len(batch_lines)}")
                for line in batch_lines:
                    try:
                        response = json.loads(line)  # Convert JSON string to dictionary

                        # Extract custom_id and ensure it follows the format "request_X"
                        custom_id = response.get("custom_id", "")
                        if custom_id.startswith("request_"):
                            row_index = int(custom_id.split("_")[1])

                            # Extract response body content
                            body = response.get("response", {}).get("body", {})
                            choices = body.get("choices", [])

                            if choices:
                                # Extract feature value from the assistant's message content
                                message_content = choices[0].get("message", {}).get("content", "{}")

                                # Ensure message_content is properly formatted as JSON
                                try:
                                    parsed_content = json.loads(message_content)  # Convert string to dictionary
                                except json.JSONDecodeError:
                                    print(
                                        f"Skipping response at index {row_index}: Invalid JSON content - {message_content}")
                                    continue

                                feature_value = parsed_content.get(feature, None)

                                if feature_value is not None:
                                    # Assign feature value to the respective model column
                                    transcript_df.at[row_index, model] = feature_value
                    except Exception as e:
                        print(f"Error processing response line: {line}. Error: {e}")

            except Exception as e:
                print(f"Error reading batch content for model {model}: {e}")
        elif model == "claude-3-7":
            for response in batch_content:
                custom_id = response.custom_id

                if custom_id.startswith("request_"):
                    row_index = int(custom_id.split("_")[1])
                    transcript_df.at[row_index, model] = json.loads(response.result.message.content[0].text).get(feature, None)

    # Save the annotated dataframe
    transcript_df.to_csv(f"{feature_dir}/atn_df.csv", index=False)
    return "annotated_df", transcript_df
