import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def readDf(filepath="data/data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\nCannot find the file at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("CSV File Unavailable!")

    print(f"\nRead Dataframe successfully. Shape: {df.shape}")
    return df

def report_missing_values(df, treatment='DELETE'):
    """Calculates, prints the missing value percentages, and returns a DF of columns with NaNs."""
    print("--- MISSING VALUES REPORT ---")
    
    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_percent = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else 0
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_counts, 
        'Percentage (%)': missing_percent
    })
    
    missing_only = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_only.empty:
        print("No missing values found in the dataset.")
    else:
        print(missing_only)
        
    justification = "Rows with missing values will be removed to ensure data quality." if treatment == 'DELETE' else "Missing values will be handled via imputation."
    print(f"\nReport Justification: '{justification}'")
    print("="*50)

def cleanQualitativeCol(df: pd.DataFrame, col: str):
    """Detects missing values for qualitative data."""
    nan_indices = df[df[col].isna()].index
    mode_val = df[col].mode()[0] if not df[col].mode().empty else None
        
    if len(nan_indices) > 0:
        print(f" - '{col}': Found {len(nan_indices)} NaNs.")
        
    return nan_indices, mode_val

def cleanQuantiativeCol(df: pd.DataFrame, col: str, threshold=3, showPlt=True):
    """Detects outliers dynamically based on skewness."""
    numeric_series = pd.to_numeric(df[col], errors='coerce')
    data = numeric_series.dropna()
    
    nan_indices = df[df[col].isna()].index
    outliersIdx = pd.Index([], dtype=int)
    
    if len(data) == 0:
        return nan_indices, outliersIdx, None 
    
    # Calculate Skewness to determine the Detection Method
    skewness = stats.skew(data)
    absSkew = abs(skewness)
    impute_val = data.mean() if absSkew < 0.5 else data.median()
    
    # Identify Outliers Based on Skewness
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
        print(f" - '{col}': Found {len(nan_indices)} NaNs, {len(outliersIdx)} outliers using {method}. (Skewness: {skewness:.2f})")

    if showPlt:
        os.makedirs("graphics", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        plt.axvline(impute_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean Value ({impute_val:.2f})')
        
        if len(outliersIdx) > 0:
            outlier_values = df.loc[outliersIdx, col]
            plt.scatter(outlier_values, np.zeros(len(outlier_values)), color='red', zorder=5, label=f'Outliers ({len(outliersIdx)})')

        plt.title(f'Distribution of {col}\nSkewness: {skewness:.2f} | Outlier Method: {method}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        output_file = f"graphics/distribution_{col}.png"
        plt.savefig(output_file)
        plt.close()
        
    return nan_indices, outliersIdx, impute_val

def exportCSV(df: pd.DataFrame, filepath="data/cleaned.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\nCleaned dataset exported successfully to {filepath}")
    return filepath

# =============================================================================
# PART 2: DATA CLEANING
# =============================================================================

def dataCleaning(filepath='data/data.csv', treatment='DELETE'):
    print("\n" + "="*50)
    print("- Part 2: Data Cleaning -")
    print("="*50)
    
    try:
        df = readDf(filepath)
        report_missing_values(df, treatment)
        
        quantiative_cols = df.select_dtypes(include=['number']).columns.tolist()
        qualitiative_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        master_drop_indices = pd.Index([])
        
        print("\n--- QUALITATIVE COLUMNS ---")
        for col in qualitiative_cols:
            nan_indices, _ = cleanQualitativeCol(df, col)
            if treatment.upper() == 'DELETE':
                master_drop_indices = master_drop_indices.union(nan_indices)
                
        print("\n--- QUANTITATIVE COLUMNS ---")
        for col in quantiative_cols:
            nan_indices, outliersIdx, _ = cleanQuantiativeCol(df, col)
            if treatment.upper() == 'DELETE':
                master_drop_indices = master_drop_indices.union(nan_indices).union(outliersIdx)

        if treatment.upper() == 'DELETE' and len(master_drop_indices) > 0:
            print("\n--- APPLYING DELETE TREATMENT ---")
            print(f"Action: Dropping a total of {len(master_drop_indices)} problematic rows (NaNs + Outliers) simultaneously.")
            df.drop(index=master_drop_indices, inplace=True)
            
        return exportCSV(df) 
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")

# =============================================================================
# PART 3: STATISTICAL ANALYSIS & INTERPRETATION
# =============================================================================

# --- Task 1: Estimation ---
def estimate(filepath='data/cleaned.csv', col='attention_span_minutes', confidence_level=0.95):
    print("\n" + "="*50)
    print("- Part 3 Task 1: Estimation -")
    print("="*50)
    
    df = pd.read_csv(filepath)
    clean_data = df[col].dropna()
    n = len(clean_data)
    
    sample_mean = clean_data.mean()
    sample_std = clean_data.std(ddof=1) 
    
    ci_lower, ci_upper = stats.t.interval(confidence=confidence_level, df=n-1, loc=sample_mean, scale=sample_std/np.sqrt(n))

    print(f"--- POINT ESTIMATES ---")
    print(f"Variable: {col}")
    print(f"Sample Size (n): {n}")
    print(f"Sample Mean: {sample_mean:.3f}")
    print(f"Sample Std Dev (s): {sample_std:.3f}")

    print(f"\n--- {int(confidence_level*100)}% CONFIDENCE INTERVAL ---")
    print(f"Result: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Q-Q Plot for Normality Check
    os.makedirs("graphics", exist_ok=True)
    plt.figure(figsize=(6, 6))
    stats.probplot(clean_data, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot - Attention Span")
    plt.ylabel("Sample Quantiles")
    plt.xlabel("Theoretical Quantiles")
    plt.savefig("graphics/qqplot_attention_span.png")
    plt.close()
    print("-> Q-Q Plot saved to 'graphics/qqplot_attention_span.png'")

# --- Task 2: Hypothesis Testing ---
def testings(filepath='data/cleaned.csv'):
    print("\n" + "="*50)
    print("- Part 3 Task 2: Hypothesis Testing -")
    print("="*50)
    
    df = pd.read_csv(filepath)
    
    # 3.2.1 ANOVA: Social Media by Gender (3 Groups)
    print("\n[Test 1: One-Way ANOVA]")
    male_data = df[df['gender'] == 'Male']['social_media_hours'].dropna()
    female_data = df[df['gender'] == 'Female']['social_media_hours'].dropna()
    other_data = df[df['gender'] == 'Other']['social_media_hours'].dropna()
    
    f_stat, p_val_anova = stats.f_oneway(male_data, female_data, other_data)
    print("-" * 60)
    print("Research question: Are there any differences in social media usage time among different gender groups?")
    print("H0: With a confidence level of 95%, there is no difference in mean social media usage across genders.")
    print("-" * 60)
    print("=== CONCLUSION ===")
    print(f"F-Statistic: {f_stat:.3f}")
    print(f"P-value: {p_val_anova:.3f}")
    if p_val_anova < 0.05:
         print(f"Reject H0 (P-value = {p_val_anova:.3f} < 0.05)")
         print("=> Significant difference exists across gender groups.")
    else:
         print(f"Fail to Reject H0 (P-value = {p_val_anova:.3f} >= 0.05)")
         print("=> No significant difference across gender groups.")

    # 3.2.2 Chi-Square: Living Area vs Brain Rot Level
    print("\n" + "="*50)
    print("[Test 2: Chi-Square Test of Independence]")
    contingency_table = pd.crosstab(df['urban_rural'], df['brain_rot_level'])
    chi2, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_table)
    
    print("=== Contingency Table ===")
    print(contingency_table)
    print("-" * 60)
    print("Research question: Is there any relations between living area and brainrot level?")
    print("H0: With a confidence level of 95%, there is no relations between living area and brainrot level.")
    print("-" * 60)
    print("=== CONCLUSION ===")
    print(f"Chi-square: {chi2:.2f}")
    print(f"P-value: {p_val_chi2:.5e}")
    if p_val_chi2 < 0.05:
        print(f"Reject H0 (P-value = {p_val_chi2:.3e} < 0.05)")
        print("=> With a confidence level of 95%, there is relations between living area and brainrot level.")
    else:
        print(f"Fail to Reject H0 (P-value = {p_val_chi2:.3e} >= 0.05)")
        print("=> With a confidence level of 95%, there is no relations between living area and brainrot level.")

    # 3.2.3 Welch's T-Test: Male vs Female Social Media Usage
    print("\n" + "="*50)
    print("[Test 3: Welch's Two-Sample T-Test]")
    t_stat_w, p_val_w = stats.ttest_ind(male_data, female_data, equal_var=False)
    print("-" * 60)
    print("Research question: Is there a significant difference in social media usage time between males and females?")
    print("H0: With a confidence level of 95%, mean social media usage for Males = Females.")
    print("-" * 60)
    print("=== CONCLUSION ===")
    print(f"T-statistic: {t_stat_w:.3f}")
    print(f"P-value: {p_val_w:.3f}")
    if p_val_w < 0.05:
         print(f"Reject H0 (P-value = {p_val_w:.3f} < 0.05)")
         print("=> Significant difference between Males and Females.")
    else:
         print(f"Fail to Reject H0 (P-value = {p_val_w:.3f} >= 0.05)")
         print("=> No significant difference.")

    # 3.2.4 One-Sample T-Test: Attention Span
    print("\n" + "="*50)
    print("[Test 4: One-Sample T-Test]")
    data_att = df['attention_span_minutes'].dropna()
    popmean_hypo = 50
    t_stat_1, p_val_1 = stats.ttest_1samp(data_att, popmean_hypo)
    print("-" * 60)
    print(f"Research question: Is the true average attention span of the population equal to {popmean_hypo} minutes?")
    print(f"H0: With a confidence level of 95%, the true average attention span = {popmean_hypo} minutes.")
    print("-" * 60)
    print("=== CONCLUSION ===")
    print(f"T-statistic: {t_stat_1:.3f}")
    print(f"P-value: {p_val_1:.5e}")
    if p_val_1 < 0.05:
         print(f"Reject H0 (P-value = {p_val_1:.3e} < 0.05)")
         print(f"=> Average attention span significantly deviates from {popmean_hypo} minutes.")
    else:
         print(f"Fail to Reject H0 (P-value = {p_val_1:.3e} >= 0.05)")
         print(f"=> Average attention span is approximately {popmean_hypo} minutes.")

# --- Task 3: Correlation & Simple Linear Regression ---
def correlation(filepath='data/cleaned.csv', col1='social_media_hours', col2='attention_span_minutes'):
    print("\n" + "="*50)
    print("- Part 3 Task 3: Correlation Analysis -")
    print("="*50)
    
    df = pd.read_csv(filepath)
    clean_data = df[[col1, col2]].dropna()
    x, y = clean_data[col1], clean_data[col2]
    n = len(clean_data)
    
    r_stat, p_value = stats.pearsonr(x, y)
    print("-" * 60)
    print(f"Research question: Is there a linear relationship between '{col1}' and '{col2}'?")
    print("H0: With a confidence level of 95%, there is no linear correlation (rho = 0).")
    print("-" * 60)
    print("=== CONCLUSION ===")
    print(f"Variables: X='{col1}', Y='{col2}' | n={n}")
    print(f"Pearson r: {r_stat:.4f} | P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        direction = "NEGATIVE" if r_stat < 0 else "POSITIVE"
        print(f"Reject H0 (P-value = {p_value:.3e} < 0.05)")
        print(f"=> Significant {direction} linear relationship exists.")
    else:
        print(f"Fail to Reject H0 (P-value = {p_value:.3e} >= 0.05)")
        print("=> No significant linear relationship.")
        
    os.makedirs("graphics", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.3, s=15, color='lightblue', edgecolor='black', linewidth=0.2)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linewidth=2.5, label=f'Trendline (r={r_stat:.2f})')
    
    plt.title(f'Correlation between Social Media Hours and Attention Span Minutes (n={n})')
    plt.xlabel('Social Media Hours')
    plt.ylabel('Attention Span Minutes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    output_file = f"graphics/correlation_scatter.png"
    plt.savefig(output_file)
    plt.close()
    print(f"-> Scatterplot saved to '{output_file}'")

def linearRegression(filepath='data/cleaned.csv', col_x='social_media_hours', col_y='attention_span_minutes'):
    print("\n" + "="*50)
    print("- Part 3 Task 3: Simple Linear Regression -")
    print("="*50)
    
    df = pd.read_csv(filepath)
    clean_data = df[[col_x, col_y]].dropna()
    x, y = clean_data[col_x], clean_data[col_y]

    res = stats.linregress(x, y)
    b1, b0 = res.slope, res.intercept
    r_squared = res.rvalue**2 
    
    print("-" * 60)
    print(f"Modeling Relationship: {col_y} ~ {col_x}")
    print("-" * 60)
    print("=== MODEL PARAMETERS ===")
    print(f"Equation: Y = {b0:.4f} + ({b1:.4f} * X)")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Slope p-value: {res.pvalue:.4e}")
    print(f"=> Interpretation: {r_squared*100:.2f}% of variance in attention span is explained by social media usage.")

    y_pred = b0 + (b1 * x)
    residuals = y - y_pred
    
    os.makedirs("graphics", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals, alpha=0.5, color='purple', edgecolor='black', s=15)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Residual Plot')
    plt.xlabel('Social Media Hours')
    plt.ylabel('Residuals')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    output_file = f"graphics/residual_plot.png"
    plt.savefig(output_file)
    plt.close()
    print(f"-> Residual plot saved to '{output_file}'")

