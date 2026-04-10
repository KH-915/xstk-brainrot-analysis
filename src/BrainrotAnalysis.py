import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest

# --------Helper Functions--------
def readDf(filepath="data/data.csv"):
    df = pd.read_csv(filepath)
    if df.empty:
        raise Exception("CSV File Unavailable!")
    
    print(f"Read Dataframe:\n{df.head()}")
    return df

def cleanEachColumn(df:pd.DataFrame, filepath="data/data.csv", col='internet_access_hours', threshold=3, debug=True):
    # Strip all periods from the string before converting
    if df.empty:
        readDf(filepath)
        
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    # Calculate Skewness
    skewness = stats.skew(data)
    absSkew = abs(skewness)
    outliersIdx = pd.Index([])
    msg = ""

    if absSkew < 0.5:
        if debug: print(f"Skewness: {skewness:.2f} (< 0.5): Using Z-Score")
        msg = "Z-Score"
        z_scores = np.abs(stats.zscore(data))
        outliersIdx = data[z_scores > threshold].index

    elif 0.5 <= absSkew <= 2.0:
        if debug: print(f"Skewness: {skewness:.2f} (0.5 - 2.0): Using IQR")

        msg = "IQR"
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lbound = Q1 - 1.5 * IQR
        rbound = Q3 + 1.5 * IQR
        
        outliersIdx = data[(data < lbound) | (data > rbound)].index
        if debug: print(f" Q1={Q1}, Q3={Q3} | Range: ({lbound:.2f}, {rbound:.2f})")

    else:
        if debug: print(f"Skewness: {skewness:.2f} (> 2.0): Using Isolation Forest")
        msg = "Isolation Forest"
        X = data.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        preds = iso_forest.fit_predict(X)
        outliersIdx = data[preds == -1].index

    if debug: print(f" Found {len(outliersIdx)} outliers.")
    return outliersIdx, msg

def exportCSV(df:pd.DataFrame, filepath="data/cleaned.csv"):
    df.to_csv(filepath)

# --------Parts--------

# Part 2
def dataCleaning(filepath, debug):
    try:
        print(f"Input filepath {filepath}, debug {debug}")
        # Read file
        df = readDf(filepath)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"Numeric Columns: {numeric_cols}")

        # Clean file
        for col in numeric_cols:
            # 1. THE FIX: Force the entire column in df to be numeric floats. 
            # This destroys the 'str' dtype and turns any text into NaN.
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 2. Now run your outlier detection
            outliers, msg = cleanEachColumn(df, col=col, debug=debug)
            print(f"- Cleaning column: {col} Using {msg}")

            if len(outliers) > 0:
                # 3. Calculate mean from the good rows
                col_mean = df.loc[~df.index.isin(outliers), col]
                # print(type(col_mean))
                col_mean = col_mean.astype(float)
                col_mean = np.mean(col_mean)
                # 4. Replace outliers with the mean
                df.loc[outliers, col] = str(col_mean)
                print(f" -> Replaced {len(outliers)} outliers with mean value: {col_mean:.2f}")
        exportCSV(df) 
        
    except Exception as e:
        print(f"Error occurs at {e}")
# Part 3
    # Task 1
def estimate(df:pd.DataFrame, confidence_level=0.95, sample_mean=0):
    # 2. Calculate components for the Confidence Interval
    sample_std = df['social_media_hours'].std()
    n = len(df['social_media_hours'].dropna()) # Sample size (excluding NaNs)
    confidence_level = 0.95

    # 3. Calculate 95% Confidence Interval using scipy
    # stats.t.interval is preferred when using sample standard deviation
    ci_lower, ci_upper = stats.t.interval(
        confidence=confidence_level, 
        df=n-1, # degrees of freedom
        loc=sample_mean, 
        scale=sample_std/np.sqrt(n) # Standard Error
    )

    print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Task 2
def oneSampleTTest(df:pd.DataFrame, target=3, col='social_media_hours', filepath="data/cleaned.csv"):
    if df.empty:
        readDf(filepath)
    t_statistic, p_value = stats.ttest_1samp(df[col].dropna(), target)
    return t_statistic, p_value

def twoSampleTTest():
    pass

def anova():
    pass

def ChiSquareTest():
    pass

def correlation():
    pass

def linearRegression():
    pass

    # Task 3
def correlation():
    pass

def regression():
    pass