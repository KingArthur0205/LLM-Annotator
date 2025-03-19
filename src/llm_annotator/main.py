# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1
import os

from argparse import ArgumentParser
from typing import List, Dict

from llm_annotator.config import ModelType
from llm_annotator.dataloader import DataLoader


def annotate(
        feature_list: List[str],
        transcript_path: str,
        sheet_source: str
):
    dataloader = DataLoader(sheet_source=sheet_source,
                            transcript_path=transcript_path)

    sheets_data = dataloader.get_features()
    transcript_df = dataloader.get_transcript()
    features = dataloader.generate_features(feature_list)


def set_working_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    os.chdir(repo_root)
    print(f"Working directory set to: {os.getcwd()}")


def main():
    set_working_dir()
    annotate(feature_list=["Mathcompetent"],
             transcript_path="./data/alltranscripts_423_clean_segmented.csv",
             sheet_source="./data/MOL Roles Features.xlsx")


if __name__ == "__main__":
    main()
