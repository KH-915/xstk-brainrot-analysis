import argparse

from src.BrainrotAnalysis import *

def run_cleaning(filepath, debug):
    """Handles the data cleaning process."""
    df = dataCleaning(filepath=filepath, debug=debug)
    return df

def run_test(mode, filepath):
    if mode==0: difInSocialMediaByGender(filepath)
    elif mode==1: livingAreaVsIncomeLevel(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing utility.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    

    clean_parser = subparsers.add_parser("clean", help="Clean the dataset and export it.")
    clean_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")
    clean_parser.add_argument("-d", "--debug", type=bool, default=False)

    test_parser = subparsers.add_parser("test", help="Test the dataset")
    test_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")
    test_parser.add_argument("-m", "--mode", type=int, default=0)

    args = parser.parse_args()

    if args.command == "clean":
        run_cleaning(args.filepath, args.debug, args.debug)
    elif args.command == "test":
        run_test(args.mode, args.filepath)
    else:
        parser.print_help()

