# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/03/16
# Version: 0.1

from argparse import ArgumentParser
from config import ModelType

from dataloader import DataLoader

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d","--data_file", type=str, default="../data/alltranscripts_423_clean_segmented.csv", help="Path of the transcript dataset(csv).")

    return parser.parse_args()

def main():
    args = parse_args()
    dataloader = DataLoader(sheet_source="1miMC8M_UkfY_3XhglTAdt9sC9H8D6d9slzC7_LgcLOc",
                            transcript_path=args.data_file)

if __name__ == "__main__":
    main()