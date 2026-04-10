import argparse

from src.BrainrotAnalysis import *

def run_cleaning(filepath, debug):
    """Handles the data cleaning process."""
    dataCleaning(filepath=filepath, debug=debug)

def run_test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing utility.")
    parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")
    parser.add_argument("-d", "--debug", type=bool, default=False)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    clean_parser = subparsers.add_parser("clean", help="Clean the dataset and export it.")
    test_parser = subparsers.add_parser("test", help="Test the dataset")

    args = parser.parse_args()

    if args.command == "clean":
        run_cleaning(args.filepath, args.debug)
    elif args.command == "test":
        run_test()
    else:
        run_cleaning()
        run_test()