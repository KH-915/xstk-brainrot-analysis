import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# --------Helper Functions--------
def readDf(filepath="data/data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find the file at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("CSV File Unavailable!")
    
    # 1. Safely force conversion on columns that look like numbers
    # 'errors="ignore"' leaves columns that are purely text (like names) alone
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # 2. Grab all resulting numeric columns and ensure they are floats
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    print(f"Read Dataframe successfully. Preview:\n{df.head()}")
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

def printDf(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.plot()
    plt.show()
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
                df.loc[outliers, col] = col_mean
                print(f" -> Replaced {len(outliers)} outliers with mean value: {col_mean:.2f}")
        printDf(df)
        exportCSV(df)
        
    except Exception as e:
        print(f"Error occurs at {e}")
# Part 3
    # Task 1
def estimate(df:pd.DataFrame, col='social_media_hours', confidence_level=0.95, sample_mean=0):
    # 2. Calculate components for the Confidence Interval
    sample_std = df[col].std()
    n = len(df[col]) # Sample size (excluding NaNs)
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

def difInSocialMediaByGender(filename='data/cleaned.csv'):
    df = pd.read_csv(filename)
    if df.empty:
        print("Please reexamine the path to your file.")
        return

    # 2. Filtering the data (gender and social media using time)
    male_data = df[df['gender'] == 'Male']['social_media_hours'].dropna()
    female_data = df[df['gender'] == 'Female']['social_media_hours'].dropna()
    other_data = df[df['gender'] == 'Other']['social_media_hours'].dropna()
    # ---------------------------------------------------------
    #Stating research question
    # ---------------------------------------------------------
    print("Research question: Is there any differences of social media using time between genders?")
    print("H0: With confidence level of 95%, evidence of differences in social media using time between genders is insufficient.") 
    print("-"*50)
    # ---------------------------------------------------------
    # Test and report
    # ---------------------------------------------------------
    # Independent 2-Sample t-test
    f_stat, p_value = stats.f_oneway(male_data, female_data, other_data)
    print("TEST STATISTICS & P-VALUE")
    print(f"Number of Male: {len(male_data)} | Mean: {male_data.mean():.2f}")
    print(f"Number of Female : {len(female_data)} | Mean: {female_data.mean():.2f}")
    print(f"Number of Other : {len(other_data)} | Mean: {other_data.mean():.2f}")
    print("-" * 50)
    print(f"F-statistic : {f_stat:.3f}")
    print(f"P-value     : {p_value:.3f}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 3. Decision & Conclusion
    # ---------------------------------------------------------
    # Checking the theory
    print("\n=== CONCLUSION ===")
    if p_value < 0.05:
        print(f"Reject H0 (P-value = {p_value:.3f} < 0.05)")
        print("Conclusion: With confidence level of 95%, there is at least one group has different social media using time.")
    else:
        print(f"Fail to Reject H0 (P-value = {p_value:.3f} >= 0.05)")
        print("Conclusion: With confidence level of 95%, evidence of differences in social media using time between genders is insufficient.")

def livingAreaVsIncomeLevel(filename='data/cleaned.csv'):
    df = pd.read_csv(filename)
    if df.empty:
        print("Please reexamine the path to your file.")
        return
    # 2. Creating contingency table
    contingency_table = pd.crosstab(df['urban_rural'], df['family_income_level'])
    print("=== Contigency Table ===")
    print(contingency_table)
    # ---------------------------------------------------------
    # Stating research question
    # ---------------------------------------------------------
    print("-"*50)
    print("Research question: Is there any relations between living area and income level?")
    print("H0: With confidence level of 95%, there is no relations between living area and income level.") 
    print("-"*50)
    # --------------------------------------------------------
    # Chi-square test run
    # --------------------------------------------------------
    print("\n=== CONCLUSION ===")
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square: {chi2:.2f}")
    print(f"P-value: {p_value:.5e}")
    if p_value < 0.05:
        print(f"Reject H0 (P-value = {p_value:.3f} < 0.05)")
        print("With confidence level of 95%, there is relations between living area and income level.")
    else:
        print(f"Reject H0 (P-value = {p_value:.3f} >= 0.05)")
        print("With confidence level of 95%, there is no relations between living area and income level.")

def correlation():
    pass

def linearRegression():
    pass

    # Task 3
def correlation():
    pass

def regression():
    pass