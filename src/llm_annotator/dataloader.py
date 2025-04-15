import gspread
import os
import pandas as pd
import numpy as np

from typing import List, Dict

import llm_annotator.utils as utils

try:
    from google.colab import drive
    from google.colab import auth
    from google.auth import default
    IN_COLAB = True
    print("Running in Google Colab.")
except ImportError:
    IN_COLAB = False
    print("Running in Local Enviornment.")


class DataLoader:
    def __init__(self,
                 sheet_source: str,
                 transcript_path: str,
                 save_dir: str = "../results"):
        if IN_COLAB:
            auth.authenticate_user()
            creds, _ = default()
            self.gc = gspread.authorize(creds)
            drive.mount('/content/drive')
        self.save_dir = save_dir
        self.transcript_df = self.__load_transcript(transcript_path)
        self.sheets_data = self.__load_features(sheet_source)
        self.features = {}

    def __load_transcript(self, transcript_path:str):
        try:
            return pd.read_csv(transcript_path)
        except:
            raise FileNotFoundError("Transcript file not found")

    def __load_features(self, source: str):
        # Check if the source is a local file
        if os.path.exists(source):
            feature_sheet = pd.ExcelFile(source)
            sheet_names = feature_sheet.sheet_names
            print("Available Sheets:", sheet_names)
            
            # Extract data from each sheet
            self.sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(source, sheet_name=sheet_name)
                    
                    # Fill in the missing Code Type
                    if "Code Type" in df.columns:
                        df["Code Type"] = df["Code Type"].replace("", None).ffill()
                    
                    self.sheets_data[sheet_name] = df
                except Exception as e:
                    raise ValueError(f"Error reading sheet '{sheet_name}': {e}")

        # Check if the source is a Google Sheet ID
        else: 
            try:
                feature_sheet = self.gc.open_by_key(source)
            except:
                raise ValueError("The provided source is neither a valid local file nor a valid Google Sheet ID.")

            sheet_names = [sheet.title for sheet in feature_sheet.worksheets()]

            # Extract the individual features from seperate sheets.
            self.sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    worksheet = feature_sheet.worksheet(sheet_name)
                    data = worksheet.get_all_values()
                except:
                    raise ValueError(f"The sheet '{sheet_name}' is not found.")
                df = pd.DataFrame(data[1:], columns=data[0])

                # Fill in the missing Code Type
                if "Code Type" in df.columns:
                    df["Code Type"] = df["Code Type"].replace("", None).ffill()
                self.sheets_data[sheet_name] = df

        return self.sheets_data


@utils.component("load_feature")
def load_features(dataloader: DataLoader):
    return "feature_df", dataloader.sheets_data


@utils.component("load_transcript")
def load_transcript(dataloader: DataLoader):
    return "transcript_df", dataloader.transcript_df


@utils.component("generate_features")
# TO-DO: Change the examples to be dynamic
def generate_features(dataloader: DataLoader, feature: str = [])\
        -> Dict:
    if dataloader.sheets_data is None:
        dataloader.__load_feautres()

    dataloader.features = {}
    for sheet_name, df in dataloader.sheets_data.items():  # Iterate over sheet names and dataframes
        for idx, row in df.iterrows():
            if "Code" in df.columns and feature == row['Code']:
                dataloader.features[feature] = {
                    "definition": row["Definition"],
                    "format": "Answer 1 if the utterance relates to the category, 0 if the utterances doesn't relate to the category.",  # Fixed typo "unkown" -> "unknown"
                    "example1": row["example1"] if "example1" in df.columns else "",
                    "example2": row["example2"] if "example2" in df.columns else "",
                    "example3": row["example3"] if "example3" in df.columns else "",
                    "nonexample1": row["nonexample1"] if "nonexample1" in df.columns else "",
                    "nonexample2": row["nonexample2"] if "nonexample2" in df.columns else "",
                    "nonexample3": row["nonexample3"] if "nonexample3" in df.columns else "",
                }
    return "feature_dict", dataloader.features



