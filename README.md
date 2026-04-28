# Quantifying Digital Overstimulation: Student "Brain Rot" Analysis

This project is an automated statistical analysis pipeline designed to investigate the impact of digital behavior—particularly social media usage—on student cognitive capacity (often colloquially referred to as "Brain Rot"). 

The pipeline performs automated data cleaning, parameter estimation, hypothesis testing, correlation analysis, and simple linear regression.

## 🛠️ Prerequisites

* **Python:** Version 3.12.8 is recommended.
* **Libraries:** All required dependencies are listed in `requirements.txt`.

## 📂 Project Structure

Before running the script, ensure your project directory is structured as follows:

```text
project_root/
│
├── data/
│   └── data.csv          # Raw input dataset (You must place this here)
│
├── requirements.txt      # Python dependencies
├── main.py               # The main execution script
└── README.md             # Project documentation
```

## 🚀 Installation & Setup

1. **Clone or download the repository** to your local machine.
2. **Place your raw dataset** (`data.csv`) into the `data/` folder.
3. **Install the required dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

To execute the full data analysis pipeline, run the main script from your terminal:

```bash
python main.py
```

## 📊 Pipeline Overview & Outputs

The script will automatically execute the following phases and generate corresponding artifacts:

1. **Data Cleaning:** * Detects missing values and dynamically applies outlier detection (Z-Score, IQR, or Isolation Forest) based on data skewness.
   * **Output:** A standardized dataset exported to `data/cleaned.csv`.
2. **Estimation (Task 1):** * Calculates point estimates and 95% Confidence Intervals for continuous variables (e.g., Attention Span).
3. **Hypothesis Testing (Task 2):** * Performs ANOVA, Chi-Square Test of Independence (Living Area vs. Brain Rot Level), Welch's T-Test, and One-Sample T-Tests.
4. **Correlation & Regression (Task 3):** * Computes Pearson correlation and constructs a Simple Linear Regression model.

### Generated Artifacts
* **`result.txt`**: A complete, step-by-step log of the terminal output containing all statistical summaries, test statistics, P-values, and automated hypothesis conclusions.
* **`graphics/`**: An automatically generated folder containing diagnostic visualizations, including:
  * Distribution histograms with mean and outlier markers.
  * Normal Q-Q plots.
  * Scatterplots with regression trendlines.
  * Residual plots for checking model assumptions.
