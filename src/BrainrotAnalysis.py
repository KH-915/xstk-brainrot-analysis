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
    """Calculates, prints the missing value percentages, and returns a DF of columns with NaNs."""
    print("--- MISSING VALUES REPORT ---")
    
    # Calculate across all columns
    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_percent = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else 0
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts, 
        'Percentage (%)': missing_percent
    })
    
    # Only show columns that actually have missing data
    missing_only = missing_df[missing_df['Missing Count'] > 0]
    
    justification = ""
    if missing_only.empty:
        print("\nNo missing values found in the dataset.")
        justification = "Data was complete; no missing value handling was required."
    else:
        print(f"\n{missing_only}")
        # Dynamic justification based on your actual cleaning choice
        if treatment == 'IMPUTE':
            justification = (f"Missing values will be handled via imputation (mean/median) "
                             f"to preserve dataset size.")
        elif treatment == 'DELETE':
            justification = (f"Rows with missing values will be removed to ensure data "
                             f"quality.")
            
    print(f"\nReport Justification: '{justification}'")
    print("="*50 + "\n")

def cleanQualitativeCol(df: pd.DataFrame, col: str):
    """
    Detects missing values for qualitative data and calculates the Mode.
    Returns: (nan_indices, mode_val)
    """
    nan_indices = df[df[col].isna()].index
    
    # Calculate the Mode (safely handle empty columns)
    mode_val = None
    if not df[col].mode().empty:
        mode_val = df[col].mode()[0]
        
    if len(nan_indices) > 0:
        print(f" - '{col}': Found {len(nan_indices)} NaNs. (Mode: '{mode_val}')")
        
    return nan_indices, mode_val


def cleanQuantiativeCol(df: pd.DataFrame, col: str, threshold=3, showPlt=True):
    """
    Detects outliers/NaNs dynamically based on skewness.
    Returns: (nan_indices, outliersIdx, impute_val)
    """
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    nan_indices = df[df[col].isna()].index
    outliersIdx = pd.Index([], dtype=int)
    impute_val = None
    
    # If column is entirely empty, just return the NaNs
    if len(data) == 0:
        return nan_indices, outliersIdx, impute_val 
    
    # Calculate Skewness to determine the "Center" and "Detection Method"
    skewness = stats.skew(data)
    absSkew = abs(skewness)
    impute_val = data.mean() if absSkew < 0.5 else data.median()
    
    # Identify Outliers
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

    if len(nan_indices) > 0 or len(outliersIdx) > 0:
        print(f" - '{col}': Found {len(nan_indices)} NaNs, {len(outliersIdx)} outliers using {method}. "
              f"(Skewness: {skewness:.2f}, Impute Val: {impute_val:.2f})")

    # ==========================================
    # NEW PLOTTING LOGIC
    # ==========================================
    if showPlt:
        plt.figure(figsize=(8, 4))
        
        # 1. Plot the histogram
        plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 2. Draw a vertical line for the chosen Impute Value (Mean/Median)
        plt.axvline(impute_val, color='red', linestyle='dashed', linewidth=2, 
                    label=f'{'Mean Value'}({impute_val:.2f})')
        
        # 3. Highlight the outliers on the graph (if any exist)
        if len(outliersIdx) > 0:
            outlier_values = df.loc[outliersIdx, col]
            # Plot tiny red dots at the bottom (y=0) for outliers
            plt.scatter(outlier_values, np.zeros(len(outlier_values)), 
                        color='red', zorder=5, label=f'Outliers ({len(outliersIdx)})')

        plt.title(f'Distribution of {col}\nSkewness: {skewness:.2f} | Outlier Method: {method}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        output_file = f"graphics/distribution_{col}.png"
        plt.savefig(output_file)
        plt.close() # Closes the figure in memory
        print(f"   -> Graph saved to {output_file}")
    return nan_indices, outliersIdx, impute_val

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
    print("\n" + "="*50)
    print("Research question: Is there any differences of social media using time between genders?")
    print("H0: With confidence level of 95%, evidence of differences in social media using time between genders is insufficient.") 
    print("="*50)
    # ---------------------------------------------------------
    # Test and report
    # ---------------------------------------------------------
    # Independent 2-Sample t-test
    f_stat, p_value = stats.f_oneway(male_data, female_data, other_data)
    print("--- TEST STATISTICS & P-VALUE ---")
    print(f"\nNumber of Male: {len(male_data)} | Mean: {male_data.mean():.2f}")
    print(f"Number of Female : {len(female_data)} | Mean: {female_data.mean():.2f}")
    print(f"Number of Other : {len(other_data)} | Mean: {other_data.mean():.2f}")

    print(f"\nF-statistic : {f_stat:.3f}")
    print(f"P-value     : {p_value:.3f}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 3. Decision & Conclusion
    # ---------------------------------------------------------
    # Checking the theory
    print("--- CONCLUSION ---")
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
    # ---------------------------------------------------------
    # Stating research question
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Research question: Is there any relations between living area and income level?")
    print("H0: With confidence level of 95%, there is no relations between living area and income level.") 
    print("="*50)
    # 2. Creating contingency table
    contingency_table = pd.crosstab(df['urban_rural'], df['family_income_level'])
    print("--- CONTIGENCY TABLE ---\n")
    print(contingency_table)
    # --------------------------------------------------------
    # Chi-square test run
    # --------------------------------------------------------
    print("-"*50)
    print("--- CONCLUSION ---")
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square: {chi2:.2f}")
    print(f"P-value: {p_value:.5e}")
    if p_value < 0.05:
        print(f"\nReject H0 (P-value = {p_value:.3f} < 0.05)")
        print("With confidence level of 95%, there is relations between living area and income level.")
    else:
        print(f"\nReject H0 (P-value = {p_value:.3f} >= 0.05)")
        print("With confidence level of 95%, there is no relations between living area and income level.")

# ----------------------------------------PARTS------------------------------------------------

# Part 2: Data Cleaning Execution
def dataCleaning(filepath='data.csv', treatment='DELETE'):
    try:
        print(f"\nInput filepath: {filepath}")
        df = readDf(filepath)
        print("\n" + "="*50)
        print("- Part 2: Data Cleaning -")
        print("="*50)
        
        # 1. Report Missing Values (Required by Rubric)
        report_missing_values(df, treatment)
        
        quantiative_cols = df.select_dtypes(include=['number']).columns.tolist()
        qualitiative_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # A master list to collect indices we want to delete later
        master_drop_indices = pd.Index([])
        
        # 2. Process Qualitative Columns
        print("--- QUALITATIVE COLUMNS ---")
        for col in qualitiative_cols:
            nan_indices, mode_val = cleanQualitativeCol(df, col)
            
            if treatment.upper() == 'IMPUTE' and len(nan_indices) > 0 and mode_val is not None:
                df.loc[nan_indices, col] = mode_val
                print(f"   -> Imputed {len(nan_indices)} NaNs with '{mode_val}'")
                
            elif treatment.upper() == 'DELETE':
                master_drop_indices = master_drop_indices.union(nan_indices)
                
        # 3. Process Quantitative Columns
        print("\n--- QUANTITATIVE COLUMNS ---")
        for col in quantiative_cols:
            nan_indices, outliersIdx, impute_val = cleanQuantiativeCol(df, col)
            
            if treatment.upper() == 'IMPUTE' and impute_val is not None:
                if len(nan_indices) > 0:
                    df.loc[nan_indices, col] = impute_val
                if len(outliersIdx) > 0:
                    df.loc[outliersIdx, col] = impute_val
                if len(nan_indices) > 0 or len(outliersIdx) > 0:
                    print(f"   -> Imputed NaNs & Outliers with {impute_val:.2f}")
                    
            elif treatment.upper() == 'DELETE':
                # Combine both NaNs and outliers into our master drop list
                master_drop_indices = master_drop_indices.union(nan_indices).union(outliersIdx)

        # 4. Final Treatment: Batch Deletion
        if treatment.upper() == 'DELETE' and len(master_drop_indices) > 0:
            print("\n--- APPLYING DELETE TREATMENT ---")
            print(f"Action: Dropping a total of {len(master_drop_indices)} dirty rows (NaNs + Outliers) simultaneously.")
            df.drop(index=master_drop_indices, inplace=True)
            
        return exportCSV(df) 
        
    except Exception as e:
        print(f"Error occurs at: {e}")
# Part 3
    # Task 1
def estimate(filepath='data/cleaned.csv', col='attention_span_minutes', confidence_level=0.95):
    """
    Constructs a Confidence Interval for the population mean of a specified column.
    """
    df = pd.read_csv(filepath)
    if df.empty:
        print("DataFrame empty or not found!")
        return
    
    print("\n" + "="*50)
    print("- Part 3 Task 1: Estimation -")
    print("="*50)
    
    # Ensure column exists
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the dataset.")
        return

    # 1. Isolate clean data
    clean_data = df[col].dropna()
    n = len(clean_data)
    
    if n < 2:
        print("Not enough data to calculate confidence intervals.")
        return

    # 2. Calculate point estimates
    sample_mean = clean_data.mean()
    sample_std = clean_data.std(ddof=1) # ddof=1 for sample standard deviation

    # 3. Calculate Confidence Interval using scipy (t-distribution)
    # The t-distribution is used because the population standard deviation is unknown.
    ci_lower, ci_upper = stats.t.interval(
        confidence=confidence_level, 
        df=n-1, # degrees of freedom
        loc=sample_mean, 
        scale=sample_std/np.sqrt(n) # Standard Error
    )

    # 4. Report the findings
    print(f"\n--- POINT ESTIMATES ---")
    print(f"Selected Variable: {col}")
    print(f"Sample Size (n): {n}")
    print(f"Sample Mean (x̄): {sample_mean:.3f}")
    print(f"Sample Std Dev (s): {sample_std:.3f}")

    print(f"\n--- {int(confidence_level*100)}% CONFIDENCE INTERVAL ---")
    print(f"Result: [{ci_lower:.3f}, {ci_upper:.3f}]")

    print("\n--- ASSUMPTIONS ---")
    print("1. Randomness and independence: Data is randomly sampled and independent.")
    print("2. Large sample size: n is large enough that the CLT ensures the sampling distribution is approximately normal.")
    print("3. No extreme outliers: The dataset is free from extreme values that disproportionately influence the mean.")

    print("\n--- PRACTICAL INTERPRETATION ---")
    print(f"We are {int(confidence_level*100)}% confident that the true population mean for '{col.replace('_', ' ')}' "
          f"lies between {ci_lower:.3f} and {ci_upper:.3f}.")
    print("="*50)

    # Task 2
def testings(filename='data/cleaned.csv', mode=0):
    if mode==0:
        print("\n" + "="*50)
        print("- Part 3 Task 2: Testings -")
        print("="*50)
        difInSocialMediaByGender(filename)
        livingAreaVsIncomeLevel(filename)
    elif mode==1:
        difInSocialMediaByGender(filename=filename)
    elif mode==2:
        livingAreaVsIncomeLevel(filename=filename)

    # Task 3
def correlation(filename='data/cleaned.csv', col1='social_media_hours', col2='attention_span_minutes'):
    """
    Calculates the Pearson correlation between two variables, performs hypothesis testing,
    and saves a scatterplot of the relationship to the 'graphics/' directory.
    """
    print("\n" + "="*50)
    print("- Part 3 Task 3: Correlation Analysis -")
    print("="*50)
    
    # 1. Load the data
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return
        
    df = pd.read_csv(filename)
    
    # Ensure columns exist
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns '{col1}' and/or '{col2}' not found in the dataset.")
        return
        
    # Isolate variables and drop missing values to ensure accurate calculation
    clean_data = df[[col1, col2]].dropna()
    x = clean_data[col1]
    y = clean_data[col2]
    n = len(clean_data)
    
    if n < 2:
        print("Not enough data to calculate correlation.")
        return
        
    # 2. Calculate Pearson Correlation (r) and P-value
    r_stat, p_value = stats.pearsonr(x, y)
    
    print(f"\nAnalyzing Correlation between '{col1}' (X) and '{col2}' (Y)")
    print(f"Sample Size (n): {n}")
    print("-" * 50)
    print(f"Pearson Correlation Coefficient (r) : {r_stat:.4f}")
    print(f"P-value                             : {p_value:.4e}")
    print("-" * 50)
    
    # 3. Hypothesis Testing & Interpretation (from rubric/PDF theory)
    print("\n--- HYPOTHESIS TEST & INTERPRETATION ---")
    print("H0: There is no linear correlation (rho = 0).")
    print("H1: There is a linear correlation (rho != 0).")
    
    if p_value < 0.05:
        print(f"\nReject H0 (p-value = {p_value:.4e} < 0.05).")
        if r_stat < 0:
            print(f"Interpretation: Significant NEGATIVE correlation (r = {r_stat:.2f}).")
            print(f"Higher '{col1}' is associated with lower '{col2}'.")
        else:
            print(f"Interpretation: Significant POSITIVE correlation (r = {r_stat:.2f}).")
            print(f"Higher '{col1}' is associated with higher '{col2}'.")
    else:
        print(f"\nFail to Reject H0 (p-value = {p_value:.4e} >= 0.05).")
        print("Interpretation: No/Weak linear correlation. The variables do not have a statistically significant linear relationship.")
        
    # 4. Draw and Save the Scatterplot (Option 1)
    # Ensure graphics directory exists
    os.makedirs("graphics", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Create the scatter points
    plt.scatter(x, y, alpha=0.5, s=15, color='skyblue', edgecolor='black', linewidth=0.2)
    
    # Create a linear trendline (y = mx + b)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linewidth=2.5, label=f'Trendline (r={r_stat:.2f})')
    
    # Formatting the graph to match the PDF example
    title_col1 = col1.replace("_", " ").title()
    title_col2 = col2.replace("_", " ").title()
    
    plt.title(f'Correlation between {title_col1} and {title_col2} (n={n})', fontsize=14)
    plt.xlabel(title_col1, fontsize=12)
    plt.ylabel(title_col2, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_file = f"graphics/correlation_{col1}_vs_{col2}.png"
    plt.savefig(output_file)
    plt.close() # Closes the figure in memory so it doesn't pop up or freeze the script
    
    print(f"\n   -> Scatterplot saved successfully to '{output_file}'")
    print("="*50)

def linearRegression(filename='data/cleaned.csv', col_x='social_media_hours', col_y='attention_span_minutes'):
    """
    Performs Simple Linear Regression, tests the slope, and generates a residual plot.
    """
    print("\n" + "="*50)
    print("- Part 3 Task 4: Simple Linear Regression -")
    print("="*50)
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return
        
    df = pd.read_csv(filename)
    
    # Isolate variables and drop missing values
    clean_data = df[[col_x, col_y]].dropna()
    x = clean_data[col_x]
    y = clean_data[col_y]
    n = len(clean_data)
    
    if n < 2:
        print("Not enough data to calculate regression.")
        return

    # 1. Estimation by Least Squares
    # linregress calculates the OLS regression line
    res = stats.linregress(x, y)
    b1 = res.slope       # Slope
    b0 = res.intercept   # Intercept
    r_squared = res.rvalue**2 # Coefficient of Determination
    p_value = res.pvalue # Hypothesis test for the slope
    
    # 2. Report the Model
    print("\n--- REGRESSION MODEL ---")
    print(f"Dependent Variable (Y) : {col_y}")
    print(f"Independent Variable (X): {col_x}")
    print(f"Equation: Y = {b0:.4f} + ({b1:.4f} * X)")
    
    print("\n--- MODEL STATISTICS ---")
    print(f"R-squared (R²): {r_squared:.4f}")
    print(f"Slope p-value : {p_value:.4e}")
    
    # 3. Interpretation
    print("\n--- INTERPRETATION ---")
    print(f"Intercept (b0): When {col_x} is 0, the expected {col_y} is {b0:.2f}.")
    
    if p_value < 0.05:
        direction = "decreases" if b1 < 0 else "increases"
        print(f"Slope (b1): Significant (p < 0.05). For every 1 unit increase in {col_x}, "
              f"{col_y} {direction} by an average of {abs(b1):.4f}.")
    else:
        print("Slope (b1): Not statistically significant (p >= 0.05). "
              "There is insufficient evidence that X predicts Y.")
              
    print(f"R-squared: {r_squared*100:.2f}% of the variance in {col_y} is explained by {col_x}.")

    # 4. Assumption Checking: Residual Analysis
    # Calculate predicted Y values and residuals (errors)
    y_pred = b0 + (b1 * x)
    residuals = y - y_pred
    
    # Plotting the Residuals
    os.makedirs("graphics", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Scatter the residuals against the independent variable
    plt.scatter(x, residuals, alpha=0.5, color='purple', edgecolor='black', s=15)
    
    # Draw a horizontal line at 0 (the ideal error)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    
    plt.title('Residual Plot (Checking Variance and Linearity Assumptions)')
    plt.xlabel(col_x.replace("_", " ").title())
    plt.ylabel('Residuals (Error: Actual - Predicted)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    output_file = f"graphics/residual_plot_{col_x}.png"
    plt.savefig(output_file)
    plt.close()
    
    print(f"\n   -> Residual plot saved successfully to '{output_file}'")
    print("="*50)

