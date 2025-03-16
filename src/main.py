# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1

from argparse import ArgumentParser
from config import ModelType
from typing import List, Dict

from dataloader import DataLoader

def annotate(
        feature_list:List[str],
        transcript_path:str,
        sheet_source:str
):

    dataloader = DataLoader(sheet_source=sheet_source,
                            transcript_path=transcript_path)
    sheets_data = dataloader.get_features()
    transcript_df = dataloader.get_transcript()
    features = dataloader.generate_features(feature_list)
