# MT2013: Brainrot Analysis & Statistical Insights

This project bridges classroom theory—specifically **Estimation** and **Correlation & Regression**—with practical analysis on real-world datasets. It implements a complete data science workflow: raw data ingestion, automated cleaning, and actionable statistical testing.

---

## 🛠 Prerequisites & Installation

To run this utility, ensure you have **Python 3.8+** installed. You will need to install the following dependencies via `pip`:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

### Required Libraries:

* `pandas`: For data manipulation and `.csv` handling.
* `numpy`: For numerical computations and mean/median calculations.
* `scipy`: For statistical tests (ANOVA, Chi-Square, T-Intervals) and skewness evaluation.
* `scikit-learn`: Specifically `IsolationForest` for advanced anomaly detection on heavily skewed data.
* `matplotlib`: For data visualization and distribution plotting.

---

## 🚀 Usage

The script uses a CLI (Command Line Interface) with two primary modes: `clean` and `test`.

### 1. Data Cleaning

Prepares the dataset by reporting missing values, handling nulls, and removing/replacing outliers dynamically based on distribution shape.

```bash
python main.py clean --filepath data/data.csv --debug True
```

* **Methodology**: Evaluates variable skewness to determine the optimal anomaly detection method: **Z-Score** (symmetric, $|S| < 0.5$), **IQR** (moderate skew, $0.5 \le |S| \le 2.0$), or **Isolation Forest** (heavy skew, $|S| > 2.0$).
* **Action**: Imputes missing values and replaces detected outliers with the **Mean** (for symmetric data) or **Median** (for skewed data) to preserve sample size and prevent biased tails.
* **Reporting**: Automatically generates formatted justification statements (Missing Values & Outlier Treatment) to be copied directly into the academic report.

### 2. Statistical Testing

Runs hypothesis tests based on the research objectives.

* **Mode 0: ANOVA Test** (Social Media Usage vs. Gender)

  ```bash
  python main.py test --mode 0
  ```
* **Mode 1: Chi-Square Test** (Living Area vs. Income Level)

  ```bash
  python main.py test --mode 1
  ```

---

## 🖥 Expected Terminal Output

When running the cleaning process, you will see a detailed breakdown of the column processing, including report-ready text:

```text
Input filepath: data/data.csv
Read Dataframe successfully. Shape: (540, 8)

==================================================
--- PART 2: MISSING VALUES REPORT ---
                       Missing Count  Percentage (%)
internet_access_hours              8            1.48
social_media_hours                12            2.22

Report Justification: 'Missing values were handled via imputation (mean/median depending on distribution skewness) to preserve the dataset size and statistical power, ensuring we maintain at least 500 observations as required.'
==================================================

--- PART 2: OUTLIER HANDLING ---

Analyzing Column: 'social_media_hours'
 - Detection Method: IQR
 - Found 14 outliers.
 - Report Justification: 'Skewness is 1.20 (between 0.5 and 2.0), indicating moderate skew. IQR was used as it is robust to skewed tails. Outliers were imputed using the Median.'
 - Action: Replaced outliers with 3.50
 - Action: Filled 12 missing values with 3.50
```

When running a test (e.g., Mode 0):

```text
Research question: Is there any differences of social media using time between genders?
H0: With confidence level of 95%, evidence of differences... is insufficient.
--------------------------------------------------
TEST STATISTICS & P-VALUE
Number of Male: 245 | Mean: 3.10
Number of Female: 255 | Mean: 3.45
--------------------------------------------------
F-statistic : 4.123
P-value     : 0.012
--------------------------------------------------
=== CONCLUSION ===
Reject H0 (P-value = 0.012 < 0.05)
Conclusion: There is at least one group with different usage time.
```

---

## 📊 Project Requirements

| Requirement      | Specification                                   |
| :--------------- | :---------------------------------------------- |
| **Dataset Size** | Minimum 500 observations                        |
| **Variables** | At least 6 (mix of quantitative & qualitative)  |
| **Format** | `.csv` source from public repositories          |
| **Deadline** | **17:00 on April 29, 2026** |
| **Submission** | Report (25 pages max), Raw/Cleaned Data, Slides |

---

## ⚖️ Academic Integrity & AI Policy

* **Similarity Limit**: Must not exceed **30%**; over 50% overlap results in a score of 1/10.
* **AI Disclosure**: All usage of AI tools (for brainstorming or editing) must be declared in the **Commitment Page** using the AI Assessment Scale.
* **Late Policy**: 50% score reduction if 1-3 days late; 0 points after 7 days.

---

**Supervising Lecturer:** Consultations available Fridays (13:30-17:00) at Room 105B4 (appointment required).