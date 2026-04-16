import argparse

from src.BrainrotAnalysis import *

def run_cleaning(filepath='data/cleaned.csv'):
    """Handles the data cleaning process."""
    df = dataCleaning(filepath)
    return df

def run_test(mode=0, filepath='data/cleaned.csv'):
    testings(filename=filepath, mode=mode)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing utility.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    

    clean_parser = subparsers.add_parser("clean", help="Clean the dataset and export it.")
    clean_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")

    test_parser = subparsers.add_parser("test", help="Test the dataset")
    test_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")
    test_parser.add_argument("-m", "--mode", type=int, default=0)

    args = parser.parse_args()

    if args.command == "clean":
        run_cleaning(args.filepath)
    elif args.command == "test":
        run_test(args.mode, args.filepath)
    else:
        filepath = run_cleaning()
        run_test(filepath=filepath)