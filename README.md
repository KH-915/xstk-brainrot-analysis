# MT2013: Brainrot Analysis & Statistical Insights

[cite_start]This project bridges classroom theory—specifically **Estimation** and **Correlation & Regression**—with practical analysis on real-world datasets[cite: 4]. [cite_start]It implements a complete data science workflow: raw data ingestion, automated cleaning, and actionable statistical testing[cite: 5].

---

## 🛠 Prerequisites & Installation

To run this utility, ensure you have **Python 3.8+** installed. You will need to install the following dependencies via `pip`:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

### Required Libraries:
* [cite_start]`pandas`: For data manipulation and `.csv` handling[cite: 17].
* `numpy`: For numerical computations and mean calculations.
* [cite_start]`scipy`: For statistical tests (ANOVA, Chi-Square, T-Intervals)[cite: 39, 49].
* [cite_start]`scikit-learn`: Specifically `IsolationForest` for advanced outlier detection[cite: 32].
* `matplotlib`: For data visualization and distribution plotting.

---

## 🚀 Usage

The script uses a CLI (Command Line Interface) with two primary modes: `clean` and `test`.

### 1. Data Cleaning
[cite_start]Prepares the dataset by handling missing values and removing/replacing outliers[cite: 31, 32].

```bash
python main.py clean --filepath data/data.csv --debug True
```
* **Methodology**: Detects outliers using **Z-Score** (for low skew), **IQR** (for moderate skew), or **Isolation Forest** (for high skew).
* **Action**: Replaces detected outliers with the column mean.

### 2. Statistical Testing
[cite_start]Runs hypothesis tests based on the research objectives[cite: 44].

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

When running the cleaning process, you should see a "sharp" breakdown of the column processing:

```text
Input filepath data/data.csv, debug True
Read Dataframe successfully. Preview:
   gender  social_media_hours  family_income_level ...
0    Male                2.5                High
...
Numeric Columns: ['social_media_hours', 'internet_access_hours']
Skewness: 0.32 (< 0.5): Using Z-Score
- Cleaning column: social_media_hours Using Z-Score
 -> Replaced 12 outliers with mean value: 3.14
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

| Requirement | Specification |
| :--- | :--- |
| **Dataset Size** | [cite_start]Minimum 500 observations [cite: 20] |
| **Variables** | [cite_start]At least 6 (mix of quantitative & qualitative) [cite: 21] |
| **Format** | [cite_start]`.csv` source from public repositories [cite: 17] |
| **Deadline** | [cite_start]**17:00 on April 29, 2026** [cite: 68] |
| **Submission** | [cite_start]Report (25 pages max), Raw/Cleaned Data, Slides [cite: 65, 91] |

---

## ⚖️ Academic Integrity & AI Policy
* [cite_start]**Similarity Limit**: Must not exceed **30%**; over 50% overlap results in a score of $1/10$[cite: 81, 83].
* [cite_start]**AI Disclosure**: All usage of AI tools (for brainstorming or editing) must be declared in the **Commitment Page** using the AI Assessment Scale[cite: 116, 119].
* **Late Policy**: 50% score reduction if 1-3 days late; [cite_start]0 points after 7 days[cite: 78, 80].

---

[cite_start]**Supervising Lecturer:** Consultations available Fridays (13:30-17:00) at Room 105B4 (appointment required)[cite: 87].