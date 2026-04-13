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

def report_missing_values(df):
    """Calculates and prints the missing value percentages for the report."""
    print("="*50)
    print("\n--- MISSING VALUES REPORT ---\n")
    
    missing_counts = df.isna().sum()
    missing_percent = (df.isna().mean() * 100).round(2)
    
    missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_percent})
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        print("No missing values found in the dataset.")
        print("Report Justification: 'Data was complete; no missing value imputation was required.'")
    else:
        print(missing_df)
        print("\nReport Justification: 'Missing values were handled via imputation (mean/median depending on distribution skewness) to preserve the dataset size and statistical power, ensuring we maintain at least 500 observations as required.'")
    print("="*50 + "\n")

def cleanEachColumn(df: pd.DataFrame, col: str, treatment='impute', threshold=3):
    """Detects and handles outliers dynamically based on skewness."""
    
    # Ensure column is numeric before proceeding
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    if len(data) == 0:
        return df # Skip if column is entirely empty or non-numeric
    
    # Calculate Skewness
    skewness = stats.skew(data)
    absSkew = abs(skewness)
    outliersIdx = pd.Index([], dtype=int)
    
    print(f"\nAnalyzing Column: '{col}'")
    
    # 1. Choose Detection Method & Imputation Value Based on Skewness
    if absSkew < 0.5:
        method = "Z-Score"
        justification = f"Skewness is {skewness:.2f} (< 0.5), indicating a symmetric distribution. Z-Score (threshold={threshold}) was used for detection. Outliers were imputed using the Mean."
        z_scores = np.abs(stats.zscore(data))
        outliersIdx = data[z_scores > threshold].index
        impute_val = data.mean() # Mean is safe for symmetric data
        
    elif 0.5 <= absSkew <= 2.0:
        method = "IQR"
        justification = f"Skewness is {skewness:.2f} (between 0.5 and 2.0), indicating moderate skew. IQR was used as it is robust to skewed tails. Outliers were imputed using the Median."
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lbound = Q1 - 1.5 * IQR
        rbound = Q3 + 1.5 * IQR
        outliersIdx = data[(data < lbound) | (data > rbound)].index
        impute_val = data.median() # Median is better for skewed data

    else:
        method = "Isolation Forest"
        justification = f"Skewness is {skewness:.2f} (> 2.0), indicating heavy skew. Isolation Forest was used for robust anomaly detection. Outliers were imputed using the Median."
        X = data.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        preds = iso_forest.fit_predict(X)
        outliersIdx = data[preds == -1].index
        impute_val = data.median() # Median is required for heavily skewed data

    print(f" - Detection Method: {method}")
    print(f" - Found {len(outliersIdx)} outliers.")
    print(f" - Report Justification: '{justification}'")

    # 2. Apply Treatment
    if len(outliersIdx) > 0:
        if treatment == 'impute':
            df.loc[outliersIdx, col] = impute_val
            print(f" - Action: Replaced outliers with {impute_val:.2f}")
        elif treatment == 'remove':
            df.drop(outliersIdx, inplace=True)
            print(f" - Action: Dropped {len(outliersIdx)} rows.")
            
    # 3. Always handle remaining NaNs (Missing Values) with the appropriate center metric
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        df[col] = df[col].fillna(impute_val)
        print(f" - Action: Filled {missing_count} missing values with {impute_val:.2f}")

    return df

def exportCSV(df: pd.DataFrame, filepath="data/cleaned.csv"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\nCleaned dataset exported successfully to {filepath}")

# --------Parts--------

# Part 2: Data Cleaning Execution
def dataCleaning(filepath):
    try:
        print(f"\nInput filepath: {filepath}")
        df = readDf(filepath)
        print("\n- PART 2 -")
        # 1. Report Missing Values (Required by Rubric)
        report_missing_values(df)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 2. Clean each numerical column
        print("--- OUTLIER HANDLING ---")
        for col in numeric_cols:
            # We pass the dataframe to update it in place, using 'impute' as the default treatment
            df = cleanEachColumn(df, col=col, treatment='impute')

        exportCSV(df)
        return df
        
    except Exception as e:
        print(f"Error occurs at: {e}")
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

    # Task 3
def correlation():
    pass

def linearRegression():
    pass

