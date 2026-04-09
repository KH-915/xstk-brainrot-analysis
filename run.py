import argparse

from src.DataCleaning import *
from src.Test import *

def run_cleaning():
    """Handles the data cleaning process."""
    try:
        filepath = None
        cols = [
            "internet_access_hours", 
            "social_media_hours", 
            "attention_span_minutes", 
            "productivity_score", 
            "academic_motivation"
        ]
        # Read file
        df = readDf(filepath)
        # Clean file
        cleaned = df
        for col in cols:
            outliers, msg = cleanDf(cleaned, col=col, debug=False)
            print(f"- Cleaning column: {col} Using {msg}")

            if len(outliers) > 0:
                cleaned = cleaned.drop(index=outliers)
                
        print(f"Remaining rows: {len(cleaned)}")
        # Export cleaned csv
        exportCSV(cleaned) 
        
    except Exception as e:
        print(f"Error occurs at {e}")

def run_test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing utility.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    clean_parser = subparsers.add_parser("clean", help="Clean the dataset and export it.")
    test_parser = subparsers.add_parser("test", help="Test the dataset")

    args = parser.parse_args()

    if args.command == "clean":
        run_cleaning()
    if args.command == "test":
        run_test()
    else:
        run_cleaning()
        run_test()