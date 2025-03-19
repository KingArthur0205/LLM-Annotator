# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1
import os
import llm_annotator.config as config
import llm_annotator.dataloader as dataloader

from argparse import ArgumentParser
from config import ModelType
from typing import List, Dict

from dataloader import DataLoader

def annotate(
        feature_list:List[str],
        transcript_path:str,
        sheet_source:str
):
    os.chdir("../..")
    print(os.getcwd())
    dataloader = DataLoader(sheet_source=sheet_source,
                            transcript_path=transcript_path)
    
    sheets_data = dataloader.get_features()
    transcript_df = dataloader.get_transcript()
    features = dataloader.generate_features(feature_list)

def main():
    annotate(feature_list=["Mathcompetent"],
         transcript_path="./data/alltranscripts_423_clean_segmented.csv",
         sheet_source="./data/MOL Roles Features.xlsx")
    
if __name__ == "__main__":
    main()