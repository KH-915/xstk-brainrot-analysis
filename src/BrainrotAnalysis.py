import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# --------Helper Functions--------
def readDf(filepath="data/data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\nCannot find the file at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("CSV File Unavailable!")

    print(f"\nRead Dataframe successfully. Shape: {df.shape}")
    print(df.head())
    return df

def report_missing_values(df, treatment='DELETE'):
    """Calculates and prints the missing value percentages for the report."""
    print("="*50)
    print("--- MISSING VALUES REPORT ---")
    
    # Calculate across all columns
    missing_counts = df.isna().sum()
    # Note: Using len(df) is safer if the dataframe is empty
    total_rows = len(df)
    missing_percent = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else 0
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts, 
        'Percentage (%)': missing_percent
    })
    
    # Only show columns that actually have missing data
    missing_only = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_only.empty:
        print("\nNo missing values found in the dataset.")
        justification = "Data was complete; no missing value handling was required."
    else:
        print(f"\n{missing_only}")
        # Dynamic justification based on your actual cleaning choice
        if treatment == 'IMPUTE':
            justification = (f"Missing values were handled via imputation (mean/median) "
                             f"to preserve dataset size (Current N={total_rows}).")
        elif treatment == 'DELETE':
            justification = (f"Rows with missing values were removed to ensure data "
                             f"quality (Current N={total_rows}).")
            
    print(f"\nReport Justification: '{justification}'")
    print("="*50 + "\n")

def cleanQualitativeCol(df: pd.DataFrame, col: str, treatment='IMPUTE'):
    """Handles missing values for qualitative data using the Mode."""
    
    # 1. Calculate the Mode (the most frequent value)
    # .mode() returns a series, so we take the first item [0]
    if df[col].mode().empty:
        return df # Skip if column is entirely empty
        
    mode_val = df[col].mode()[0]
    nan_count = df[col].isna().sum()
    
    print(f"\nAnalyzing Qualitative Column: '{col}'")
    
    if treatment.upper() == 'IMPUTE':
        if nan_count > 0:
            df[col] = df[col].fillna(mode_val)
            print(f" - Action: Filled {nan_count} NaNs with Mode: '{mode_val}'")
            
    elif treatment.upper() == 'DELETE':
        if nan_count > 0:
            df.dropna(subset=[col], inplace=True)
            print(f" - Action: Deleted {nan_count} rows with missing value")
            
    return df

def cleanQuantiativeCol(df: pd.DataFrame, col: str, treatment='DELETE', threshold=3):
    """
    Detects and handles outliers/NaNs dynamically based on skewness.
    Treatments: 
    - 'IMPUTE': Replaces outliers and NaNs with the calculated center (Mean/Median).
    - 'DELETE': Drops rows containing outliers or NaNs in this column.
    """
    # 1. Ensure column is numeric and isolate valid data for stats calculation
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    if len(data) == 0:
        # If the whole column is NaN, we can't calculate stats. 
        # If treatment is DELETE, we clear the column.
        if treatment == 'DELETE':
            df.dropna(subset=[col], inplace=True)
        return df 
    
    # Calculate Skewness to determine the best "Center" and "Detection Method"
    skewness = stats.skew(data)
    absSkew = abs(skewness)
    outliersIdx = pd.Index([], dtype=int)
    
    # Determine the center metric (Mean for normal, Median for skewed)
    impute_val = data.mean() if absSkew < 0.5 else data.median()
    
    print(f"\nAnalyzing Column: '{col}' (Skewness: {skewness:.2f})")
    
    # 2. Identify Outliers
    if absSkew < 0.5:
        method = "Z-Score"
        z_scores = np.abs(stats.zscore(data))
        outliersIdx = data[z_scores > threshold].index
        
    elif 0.5 <= absSkew <= 2.0:
        method = "IQR"
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        outliersIdx = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))].index

    else:
        method = "Isolation Forest"
        X = data.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        preds = iso_forest.fit_predict(X)
        outliersIdx = data[preds == -1].index

    # 3. Apply Treatment for Outliers and NaNs
    nan_indices = df[df[col].isna()].index
    
    if treatment == 'IMPUTE':
        # Replace Outliers
        if len(outliersIdx) > 0:
            df.loc[outliersIdx, col] = impute_val
            print(f" - Outliers: Replaced {len(outliersIdx)} values using {method} with {impute_val:.2f}")
        
        # Replace NaNs
        if len(nan_indices) > 0:
            df[col] = df[col].fillna(impute_val)
            print(f" - NaNs: Filled {len(nan_indices)} missing values with {impute_val:.2f}")

    elif treatment == 'DELETE':
        # Combine outlier indices and NaN indices, then drop
        total_to_drop = outliersIdx.union(nan_indices)
        if len(total_to_drop) > 0:
            df.drop(total_to_drop, inplace=True)
            print(f" - {len(outliersIdx)} outliers found using {method}")
            print(f" - Action: Deleted {len(total_to_drop)} rows (Outliers + NaNs)")

    return df

def exportCSV(df: pd.DataFrame, filepath="data/cleaned.csv"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\nCleaned dataset exported successfully to {filepath}")
    return filepath

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
    # Stating research question
    # ---------------------------------------------------------
    print("\nResearch question: Is there any differences of social media using time between genders?")
    print("H0: With confidence level of 95%, evidence of differences in social media using time between genders is insufficient.") 
    print("="*50)
    # ---------------------------------------------------------
    # Test and report
    # ---------------------------------------------------------
    # Independent 2-Sample t-test
    f_stat, p_value = stats.f_oneway(male_data, female_data, other_data)
    print("\n--- TEST STATISTICS & P-VALUE ---")
    print(f"\nNumber of Male: {len(male_data)} | Mean: {male_data.mean():.2f}")
    print(f"Number of Female : {len(female_data)} | Mean: {female_data.mean():.2f}")
    print(f"Number of Other : {len(other_data)} | Mean: {other_data.mean():.2f}")
    print("=" * 50)
    print(f"F-statistic : {f_stat:.3f}")
    print(f"P-value     : {p_value:.3f}")
    print("=" * 50)

    # ---------------------------------------------------------
    # 3. Decision & Conclusion
    # ---------------------------------------------------------
    # Checking the theory
    print("\n--- CONCLUSION ---")
    if p_value < 0.05:
        print(f"\nReject H0 (P-value = {p_value:.3f} < 0.05)")
        print("Conclusion: With confidence level of 95%, there is at least one group has different social media using time.")
    else:
        print(f"\nFail to Reject H0 (P-value = {p_value:.3f} >= 0.05)")
        print("Conclusion: With confidence level of 95%, evidence of differences in social media using time between genders is insufficient.")

def livingAreaVsIncomeLevel(filename='data/cleaned.csv'):
    df = pd.read_csv(filename)
    if df.empty:
        print("Please reexamine the path to your file.")
        return
    # 2. Creating contingency table
    contingency_table = pd.crosstab(df['urban_rural'], df['family_income_level'])
    print("\n--- Contigency Table ---\n")
    print(contingency_table)
    # ---------------------------------------------------------
    # Stating research question
    # ---------------------------------------------------------
    print("="*50)
    print("Research question: Is there any relations between living area and income level?")
    print("H0: With confidence level of 95%, there is no relations between living area and income level.") 
    print("="*50)
    # --------------------------------------------------------
    # Chi-square test run
    # --------------------------------------------------------
    print("\n--- CONCLUSION ---")
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square: {chi2:.2f}")
    print(f"P-value: {p_value:.5e}")
    if p_value < 0.05:
        print(f"Reject H0 (P-value = {p_value:.3f} < 0.05)")
        print("With confidence level of 95%, there is relations between living area and income level.")
    else:
        print(f"Reject H0 (P-value = {p_value:.3f} >= 0.05)")
        print("With confidence level of 95%, there is no relations between living area and income level.")

# ----------------------------------------PARTS------------------------------------------------

# Part 2: Data Cleaning Execution
def dataCleaning(filepath, treatment='DELETE'):
    try:
        print(f"\nInput filepath: {filepath}")
        df = readDf(filepath)
        print("\n- Part 2: Data Cleaning -\n")
        # 1. Report Missing Values (Required by Rubric)
        report_missing_values(df)
        
        quantiative_cols = df.select_dtypes(include=['number']).columns.tolist()
        qualitiative_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        # 2. Clean each numerical column
        print("\n--- OUTLIER HANDLING ---\n")
        print("\n--- Qualitiative Columns ---")
        for col in qualitiative_cols:
            df = cleanQualitativeCol(df=df, col=col, treatment=treatment)
        print("\n--- Quantitiative Columns ---")
        for col in quantiative_cols:
            df = cleanQuantiativeCol(df, col=col, treatment=treatment)

        return exportCSV(df) 
        
    except Exception as e:
        print(f"Error occurs at: {e}")
# Part 3
    # Task 1
def estimate(df:pd.DataFrame, col='social_media_hours', confidence_level=0.95, sample_mean=0):
    print("\n- Part 3 Task 1: Estimation -")
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
def testings(filename='data/cleaned.csv', mode=0):
    if mode==0:
        print("\n- Part 3 Task 2: Testings -")
        difInSocialMediaByGender(filename)
        livingAreaVsIncomeLevel(filename)
    elif mode==1:
        difInSocialMediaByGender(filename=filename)
    elif mode==2:
        livingAreaVsIncomeLevel(filename=filename)

    # Task 3
def correlation():
    pass

def linearRegression():
    pass

