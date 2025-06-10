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
        batch_dir = os.path.join(batch_dir, timestamp)
    else:
        # Create "results" directory if it does not exist
        batch_dir = utils.create_batch_dir(save_dir=save_dir, feature=feature, timestamp=timestamp)
    transcript_df = transcript_df.copy()

    for model, batch_content in batch_results.items():
        print(f"Processing {model} results...")
        
        if model == "gpt-4o":
            try:
                # Ensure batch_content is a string and split into lines (each line is a JSON object)
                batch_lines = batch_content.strip().split("\n")
                print(f"Found {len(batch_lines)} GPT-4o response lines")
                
                for line_idx, line in enumerate(batch_lines):
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
                            except json.JSONDecodeError as e:
                                print(
                                    f"Skipping response at line {line_idx}: Invalid JSON content - {message_content}")
                                print(f"JSON decode error: {e}")
                                continue

                            result_log_probs = []
                            for prob in log_probs:
                                if prob['token'] in ("0", "1"):
                                    result_log_probs.append(round(prob['logprob'], 3))

                            index = 0
                            annotations_processed = 0
                            for utt_id, value in parsed_content.items():
                                matching_row = transcript_df['uttid'] == utt_id
                                if matching_row.any():
                                    # Use feature name instead of model name for the column
                                    transcript_df.loc[matching_row, feature] = value
                                    transcript_df.loc[matching_row, f"{model}_logprob"] = result_log_probs[index] if index < len(result_log_probs) else None
                                    annotations_processed += 1
                                else:
                                    print(f"Warning: No matching row found for uttid {utt_id}")
                                index += 1
                            print(f"Processed {annotations_processed} annotations from GPT-4o response {line_idx}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_idx}: {e}")
                        print(f"Line content: {line}")
                    except Exception as e:
                        print(f"Error processing response line {line_idx}: {line}. Error: {e}")
            except Exception as e:
                print(f"Error reading batch content for model {model}: {e}")
                
        elif model == "claude-3-7":
            print(f"Found {len(batch_content)} Claude-3-7 responses")
            
            if not batch_content:
                print("Warning: No Claude-3-7 responses found in batch_content")
                continue
                
            total_annotations_processed = 0
            for response_idx, response in enumerate(batch_content):
                try:
                    print(f"Processing Claude-3-7 response {response_idx + 1}/{len(batch_content)}")
                    
                    # Check if response has the expected structure
                    if not hasattr(response, 'result'):
                        print(f"Error: Response {response_idx} missing 'result' attribute")
                        print(f"Response structure: {type(response)}")
                        continue
                        
                    if not hasattr(response.result, 'message'):
                        print(f"Error: Response {response_idx} missing 'message' in result")
                        continue
                        
                    if not hasattr(response.result.message, 'content') or not response.result.message.content:
                        print(f"Error: Response {response_idx} missing or empty 'content' in message")
                        continue
                    
                    response_text = response.result.message.content[0].text
                    print(f"Raw response text: {response_text[:200]}...")  # Print first 200 chars
                    
                    # Handle JSON code blocks
                    if "```json" in response_text:
                        extracted_json = extract_json_code_block(response_text)
                        if extracted_json is None:
                            print(f"Error: Could not extract JSON from code block in response {response_idx}")
                            continue
                        response_text = extracted_json
                    
                    # Clean up the response text
                    response_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                    print(f"Cleaned response text: {response_text}")
                    
                    try:
                        parsed_response = json.loads(response_text)
                        print(f"Successfully parsed JSON with {len(parsed_response)} items")
                    except json.JSONDecodeError as e:
                        print(f"Error: Could not parse JSON in response {response_idx}: {e}")
                        print(f"Response text that failed to parse: {response_text}")
                        continue

                    annotations_processed = 0
                    for utt_id, value in parsed_response.items():
                        matching_row = transcript_df['uttid'] == utt_id
                        if matching_row.any():
                            # Use feature name instead of model name for the column
                            transcript_df.loc[matching_row, feature] = value
                            annotations_processed += 1
                        else:
                            print(f"Warning: No matching row found for uttid {utt_id}")
                    
                    print(f"Processed {annotations_processed} annotations from Claude-3-7 response {response_idx}")
                    total_annotations_processed += annotations_processed
                    
                except AttributeError as e:
                    print(f"Error: Response structure issue in response {response_idx}: {e}")
                    print(f"Response type: {type(response)}")
                    if hasattr(response, '__dict__'):
                        print(f"Response attributes: {list(response.__dict__.keys())}")
                except json.JSONDecodeError as e:
                    print(f"Error: JSON parsing failed in response {response_idx}: {e}")
                except Exception as e:
                    print(f"Error processing Claude-3-7 response {response_idx}: {e}")
                    print(f"Response type: {type(response)}")
                    continue
            
            print(f"Total Claude-3-7 annotations processed: {total_annotations_processed}")

    # Save the annotated dataframe
    if batch_results:
        print(f"Saving results to: {batch_dir}")
        transcript_df.to_csv(f"{batch_dir}/atn_df.csv", index=False)
        print(f"Results saved. Dataframe shape: {transcript_df.shape}")
        print(f"Columns in dataframe: {list(transcript_df.columns)}")
    return "annotated_df", transcript_df
