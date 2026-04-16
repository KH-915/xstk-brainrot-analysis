import argparse

from src.BrainrotAnalysis import *

def run_cleaning(filepath='data/data.csv'):
    """Handles the data cleaning process."""
    df = dataCleaning(filepath)
    return df

def run_test(mode=0, filepath='data/cleaned.csv'):
    testings(filename=filepath, mode=mode)

def run_estimate(filepath='data/cleaned.csv', col='attention_span_minutes', confidence_level=0.95):
    estimate(filepath=filepath, col=col, confidence_level=confidence_level)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing utility.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    

    clean_parser = subparsers.add_parser("clean", help="Clean the dataset and export it.")
    clean_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")

    estimate_parser = subparsers.add_parser("estimate", help="Construct a 95 percent confidence interval")

    test_parser = subparsers.add_parser("test", help="Test the dataset")
    test_parser.add_argument("-f", "--filepath", type=str, default="data/data.csv")
    test_parser.add_argument("-m", "--mode", type=int, default=0)

    correlation_parser = subparsers.add_parser("correlation")

    regression_parser = subparsers.add_parser("regression")

    args = parser.parse_args()

    if args.command == "clean":
        run_cleaning(args.filepath)
    elif args.command == "test":
        run_test(args.mode, args.filepath)
    elif args.command == "estimate":
        run_estimate()
    elif args.command == "correlation":
        correlation()
    elif args.command == "regression":
        linearRegression()
    else:
        filepath = run_cleaning()
        run_test(filepath=filepath)