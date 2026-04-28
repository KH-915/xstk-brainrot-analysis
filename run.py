import numpy as np
from src.BrainrotAnalysis import dataCleaning, estimate, testings, correlation, linearRegression
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    RAW_DATA_PATH = 'data/data.csv'
    CLEAN_DATA_PATH = 'data/cleaned.csv'
    
    # Execute Pipeline
    dataCleaning(filepath=RAW_DATA_PATH, treatment='DELETE')
    estimate(filepath=CLEAN_DATA_PATH)
    testings(filepath=CLEAN_DATA_PATH)
    correlation(filepath=CLEAN_DATA_PATH)
    linearRegression(filepath=CLEAN_DATA_PATH)