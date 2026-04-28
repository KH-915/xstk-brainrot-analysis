import numpy as np
import sys
from src.BrainrotAnalysis import dataCleaning, estimate, testings, correlation, linearRegression

# =============================================================================
# UTILITY: TERMINAL REDIRECTION
# =============================================================================
class Logger(object):
    """Redirects terminal output to both the console and a text file."""
    def __init__(self, filename="result.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Start logging to terminal and result.txt
    sys.stdout = Logger("result.txt")
    
    np.random.seed(42)
    
    # Path settings
    RAW_DATA_PATH = 'data/data.csv'
    CLEAN_DATA_PATH = 'data/cleaned.csv'
    
    # Run the Pipeline
    dataCleaning(filepath=RAW_DATA_PATH)
    estimate(filepath=CLEAN_DATA_PATH)
    testings(filepath=CLEAN_DATA_PATH)
    
    print("\n" + "="*50)
    print("Execution complete. All logs saved to result.txt.")
    print("="*50)