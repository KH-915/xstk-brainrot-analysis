import pandas as pd
import numpy as np
from scipy.stats import skew, zscore, quantile
from sklearn.ensemble import IsolationForest

def readDf(filepath="data/data.csv"):
    df = pd.read_csv(filepath)
    if df.empty:
        raise Exception("CSV File Unavailable!")
    print(f"Read Dataframe:\n{df.head()}")
    return df

def cleanDf(df:pd.DataFrame, col='internet_access_hours', threshold=3, debug=True):
    # Strip all periods from the string before converting
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    # Calculate Skewness
    skewness = skew(data)
    absSkew = abs(skewness)
    outliersIdx = pd.Index([])
    msg = ""

    if absSkew < 0.5:
        if debug:
            print(f"Skewness: {skewness:.2f} (< 0.5): Using Z-Score")
        msg = "Z-Score"
        z_scores = np.abs(zscore(data))
        outliersIdx = data[z_scores > threshold].index

    elif 0.5 <= absSkew <= 2.0:
        if debug:
            print(f"Skewness: {skewness:.2f} (0.5 - 2.0): Using IQR")

        msg = "IQR"
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lbound = Q1 - 1.5 * IQR
        rbound = Q3 + 1.5 * IQR
        
        outliersIdx = data[(data < lbound) | (data > rbound)].index
        if debug:
            print(f" Q1={Q1}, Q3={Q3} | Range: ({lbound:.2f}, {rbound:.2f})")

    else:
        if debug:
            print(f"Skewness: {skewness:.2f} (> 2.0): Using Isolation Forest")
        msg = "Isolation Forest"
        X = data.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        preds = iso_forest.fit_predict(X)
        outliersIdx = data[preds == -1].index

    if debug:
        print(f" Found {len(outliersIdx)} outliers.")
    return outliersIdx, msg

def exportCSV(df:pd.DataFrame, filepath="data/cleaned.csv"):
    df.to_csv(filepath)

