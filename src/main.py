# Author: Arthur Pan(The University of Edinburgh)
# Date: 2025/02/25
# Version: 0.1

from argparse import ArgumentParser
from config import ModelType

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d","--data_file", type=str, default="../data/alltranscripts_423_clean_segmented.csv", help="Path of the transcript dataset(csv).")

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

if __name__ == "__main__":
    main()