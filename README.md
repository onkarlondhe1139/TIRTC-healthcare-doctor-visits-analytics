# 🏥 Healthcare Analytics for Doctor Visits

<p align="center">

**An End-to-End Healthcare Data Analytics & Machine Learning Project**

Analyzing demographic, financial, health, and healthcare-support factors associated with doctor visits using Python, Exploratory Data Analysis, Machine Learning, and an interactive Streamlit dashboard.

</p>

---

## 📌 Project Overview

Healthcare utilization varies considerably across individuals. Understanding the factors associated with the frequency of doctor visits can help identify important patterns in patient-related and healthcare-support data.

This project performs an end-to-end analysis of a healthcare dataset containing **5,190 records and 13 variables**. The analysis investigates the relationship between doctor visits and factors such as age, gender, income, illness, reduced activity, general health, chronic conditions, and healthcare support.

The project combines:

* Data cleaning and quality assessment
* Exploratory Data Analysis (EDA)
* Statistical and group-based analysis
* Data visualization
* Regression-based machine learning
* Model evaluation
* Interactive Streamlit dashboard
* Automated analysis outputs
* Professional project documentation

### Core Research Question

> **Why do some people visit a doctor more often than others?**

The project focuses on identifying **statistical patterns and associations** in the dataset rather than making medical diagnoses or causal claims.

---

# 🎯 Project Objectives

The primary objectives of this project are to:

1. Understand the structure and quality of the healthcare dataset.
2. Analyze the distribution of doctor visits.
3. Study demographic patterns related to doctor visits.
4. Examine the relationship between illness and doctor visits.
5. Analyze the association between reduced activity and doctor visits.
6. Investigate the relationship between health indicators and doctor visits.
7. Compare doctor visits across chronic-condition groups.
8. Examine the role of healthcare-support variables.
9. Develop machine-learning models for predicting doctor visits.
10. Compare model performance using standard regression metrics.
11. Build an interactive dashboard for healthcare data exploration.
12. Generate reproducible analysis outputs and reports.

---

# 🧩 Problem Statement

Doctor visits can vary between individuals due to differences in health conditions, illness levels, demographic characteristics, economic background, and access to healthcare support.

The objective of this project is to analyze these variables systematically and determine which characteristics are associated with differences in the number of doctor visits.

The project therefore addresses the following analytical questions:

* How are doctor visits distributed across the population?
* Is there an observable relationship between age and doctor visits?
* How do doctor visits differ by gender?
* Does the number of illnesses show an association with doctor visits?
* Is reduced daily activity associated with more doctor visits?
* How are health indicators related to healthcare utilization?
* How do chronic-condition groups differ in doctor visits?
* How do healthcare-support categories compare?
* Can doctor visits be estimated using machine-learning models?

---

# 📊 Dataset Description

The dataset contains **5,190 records and 13 columns**.

The variables can be grouped into five major categories.

| Category              | Variables                                              |
| --------------------- | ------------------------------------------------------ |
| Personal Information  | `gender`, `age`                                        |
| Financial Information | `income`                                               |
| Health Information    | `illness`, `reduced`, `health`, `nchronic`, `lchronic` |
| Healthcare Support    | `private`, `freepoor`, `freerepat`                     |
| Doctor Visits         | `visits`                                               |
| Record Information    | `Unnamed: 0`                                           |

---

## 📋 Feature Description

| Feature      | Description                                                              | Role                   |
| ------------ | ------------------------------------------------------------------------ | ---------------------- |
| `visits`     | Number of doctor visits during the covered period                        | Target                 |
| `gender`     | Gender of the individual                                                 | Predictor              |
| `age`        | Age of the individual                                                    | Predictor              |
| `income`     | Income level / financial background                                      | Predictor              |
| `illness`    | Number of illnesses or health problems                                   | Predictor              |
| `reduced`    | Number of days normal activities were reduced because of health problems | Predictor              |
| `health`     | Numerical indicator representing general health condition                | Predictor              |
| `private`    | Private healthcare / insurance coverage                                  | Predictor              |
| `freepoor`   | Free healthcare support based on financial need                          | Predictor              |
| `freerepat`  | Special repatriation-related healthcare support                          | Predictor              |
| `nchronic`   | Indicator of a chronic health condition                                  | Predictor              |
| `lchronic`   | Indicator of a long-term chronic health condition                        | Predictor              |
| `Unnamed: 0` | Record identifier                                                        | Excluded from modeling |

> `Unnamed: 0` is treated as an identifier rather than a healthcare measurement and is therefore excluded from the machine-learning features.

---

# 🔬 Analytical Methodology

The project follows a structured data-science workflow:

```text
                 Healthcare Dataset
                         │
                         ▼
                Data Loading & Inspection
                         │
                         ▼
                Data Quality Assessment
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
              Exploratory Data Analysis
                         │
                         ▼
             Statistical / Group Analysis
                         │
                         ▼
                  Data Visualization
                         │
                         ▼
               Machine Learning Models
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
              Interactive Dashboard
                         │
                         ▼
                  Insights & Reports
```

---

# 🧹 Data Preparation & Quality Assessment

The dataset was inspected before performing analytical and machine-learning operations.

The preparation process included:

* Dataset loading using Pandas
* Shape and structure inspection
* Column identification
* Data-type verification
* Missing-value analysis
* Duplicate checking
* Numerical-variable inspection
* Categorical-variable inspection
* Identifier handling
* Feature-target separation
* Machine-learning preprocessing

### Data Quality Result

The supplied dataset contains **no missing values** based on the implemented quality analysis.

The record identifier `Unnamed: 0` was not used as a predictive feature.

---

# 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution and relationships present in the dataset.

## 1. Doctor Visit Distribution

![Doctor Visit Distribution](https://raw.githubusercontent.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/main/Output/01_visits_distribution.png)

The doctor-visit variable is examined to understand how frequently individuals reported visiting a doctor.

The distribution is **right-skewed**, with many observations having zero visits and comparatively fewer observations having higher numbers of visits.

---

## 2. Age Distribution

![Age Distribution](https://github.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/blob/main/Output/02_age_distribution.png)

The age distribution provides an overview of the population represented in the dataset and helps identify the range and concentration of observations.

---

## 3. Doctor Visits by Gender

![Doctor Visits by Gender](https://github.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/blob/main/Output/03_visits_by_gender.png)

Doctor-visit patterns are compared across gender groups to identify differences in the observed distribution and average utilization.

These differences represent dataset-level associations and should not be interpreted as causal effects.

---

## 4. Reduced Activity vs Doctor Visits

![Reduced Activity vs Visits](https://github.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/blob/main/Output/04_reduced_vs_visits.png)

This analysis examines the relationship between the number of days in which normal activities were reduced due to health problems and doctor visits.

Among the numerical variables analyzed, `reduced` shows the strongest observed association with `visits`.

---

## 5. Illness vs Doctor Visits

![Illness vs Visits](https://github.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/blob/main/Output/05_illness_vs_visits.png)

This visualization examines whether the number of reported illnesses is associated with the frequency of doctor visits.

The analysis indicates that `illness` is also meaningfully associated with doctor visits in the supplied dataset.

---

## 6. Numerical Correlation Analysis

![Correlation Heatmap](https://github.com/onkarlondhe1139/TIRTC-healthcare-doctor-visits-analytics/blob/main/Output/06_correlation_heatmap.png)

The correlation heatmap provides an overview of relationships among numerical variables.

The analysis particularly examines the relationship between:

* `visits`
* `age`
* `income`
* `illness`
* `reduced`
* `health`

The correlation results are used for exploratory interpretation and do not establish causation.

---

# 📊 Statistical & Group Analysis

In addition to numerical relationships, the project performs group-based comparisons.

### Demographic Analysis

* Gender-wise doctor visits
* Age distribution

### Financial Analysis

* Income-related patterns

### Health Analysis

* Illness
* General health
* Reduced activity
* Chronic health conditions
* Long-term chronic conditions

### Healthcare Support Analysis

* Private healthcare coverage
* Free healthcare support
* Repatriation-related support

These comparisons help identify differences between groups within the dataset.

---

# 🤖 Machine Learning

Machine learning is used to model the number of doctor visits as a regression problem.

### Target Variable

```text
visits
```

### Problem Type

```text
Regression
```

The objective is to estimate the numerical value of doctor visits using available demographic, financial, health, and healthcare-support variables.

---

# 🧠 Machine Learning Models

Two regression algorithms were implemented.

## 1. Ridge Regression

Ridge Regression is a regularized linear regression technique.

It was included to provide a relatively simple and interpretable baseline model while reducing the effect of multicollinearity through regularization.

---

## 2. Random Forest Regression

Random Forest Regression is an ensemble machine-learning method that combines multiple decision trees.

It was included to capture potentially non-linear relationships between the predictor variables and doctor visits.

---

# ⚙️ Machine Learning Pipeline

The implemented pipeline follows these steps:

```text
Raw Dataset
     │
     ▼
Feature / Target Separation
     │
     ▼
80/20 Train-Test Split
     │
     ▼
Numerical Preprocessing
     │
     ├── Imputation
     └── Standardization
     
Categorical Preprocessing
     │
     ├── Imputation
     └── One-Hot Encoding
     
     ▼
Model Training
     │
     ├── Ridge Regression
     └── Random Forest Regression
     
     ▼
Predictions
     │
     ▼
Model Evaluation
```

The train-test split uses:

```python
random_state = 42
```

---

# 📏 Model Evaluation Metrics

Three standard regression metrics were used.

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

**Lower MAE indicates smaller prediction errors.**

### Root Mean Squared Error — RMSE

Measures prediction error while giving greater weight to larger errors.

**Lower RMSE indicates better performance.**

### R² Score

Measures the proportion of variation in the target variable explained by the model.

Higher values indicate greater explanatory performance for the evaluated test data.

---

# 📊 Model Performance

The models produced the following results on the test set:

| Model                    |    MAE |   RMSE |     R² |
| ------------------------ | -----: | -----: | -----: |
| Ridge Regression         | 0.4202 | 0.8348 | 0.2118 |
| Random Forest Regression | 0.4193 | 0.8589 | 0.1656 |

### Interpretation

The two models produced relatively similar MAE values, while the Ridge Regression model produced the higher R² and lower RMSE in this particular experiment.

The results should be interpreted specifically for the supplied dataset and implemented train-test configuration rather than generalized to other healthcare populations.

---

# 💡 Key Findings

The analysis produced several important observations:

### Doctor Visit Distribution

Doctor visits are highly concentrated toward lower values, with many records showing zero visits.

### Health-Related Variables

Health-related variables show meaningful associations with doctor visits.

### Reduced Activity

`reduced` demonstrates the strongest observed numerical association with `visits` among the analyzed numerical variables.

### Illness

`illness` also demonstrates a noticeable association with doctor visits.

### Gender

The distribution of doctor visits differs across gender groups in the dataset.

### Chronic Conditions

Individuals can be compared based on chronic and long-term chronic-condition indicators to examine differences in doctor-visit patterns.

### Healthcare Support

The healthcare-support variables allow comparisons between individuals with different types of healthcare coverage or support.

> **Important:** These findings describe statistical patterns in the supplied dataset. They do not establish that one variable causes another.

---

# 🖥️ Interactive Streamlit Dashboard

The project includes an interactive dashboard developed using **Streamlit**.

The dashboard allows users to explore the healthcare dataset without directly modifying the source code.

## Dashboard Features

### KPI Summary

The dashboard displays:

* Total number of records
* Average doctor visits
* Average number of illnesses
* Average reduced-activity days

### Interactive Filters

Users can filter the analysis using:

* Gender
* Chronic condition status
* Long-term chronic condition status

### Visual Analysis

The dashboard provides interactive views of:

* Doctor visits
* Demographic patterns
* Health-related variables
* Illness
* Reduced activity
* Healthcare-support categories

### Data Export

Filtered data can be downloaded as a CSV file directly from the dashboard.

---

# 📂 Project Structure

```text
healthcare_doctor_visits_project/
│
├── data/
│   └── doctor_visits.csv
│
├── outputs/
│   ├── 01_visits_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_visits_by_gender.png
│   ├── 04_reduced_vs_visits.png
│   ├── 05_illness_vs_visits.png
│   ├── 06_correlation_heatmap.png
│   ├── data_quality.csv
│   ├── descriptive_statistics.csv
│   ├── group_analysis.txt
│   ├── model_comparison.csv
│   └── test_predictions.csv
│
├── separate_images/
│   ├── 01_visits_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_visits_by_gender.png
│   ├── 04_reduced_vs_visits.png
│   ├── 05_illness_vs_visits.png
│   └── 06_correlation_heatmap.png
│
├── src/
│   ├── analysis.py
│   └── app.py
│
├── PROJECT_REPORT.md
├── README.md
├── README_GITHUB.md
├── PROJECT_STRUCTURE.txt
├── requirements.txt
│
├── Healthcare_Doctor_Visits_Project_Report_with_Images.docx
└── Healthcare_Doctor_Visits_Project_Report_with_Images.pdf
```

---

# 🛠️ Technology Stack

| Technology       | Purpose                            |
| ---------------- | ---------------------------------- |
| **Python**       | Core programming language          |
| **Pandas**       | Data manipulation and analysis     |
| **NumPy**        | Numerical computation              |
| **Matplotlib**   | Data visualization                 |
| **Scikit-learn** | Machine learning and preprocessing |
| **Streamlit**    | Interactive analytics dashboard    |
| **OpenPyXL**     | Spreadsheet-related processing     |
| **Git**          | Version control                    |
| **GitHub**       | Project hosting and collaboration  |

---

# 💻 System Requirements

## Minimum Requirements

* Python **3.10 or later**
* 4 GB RAM or more
* Internet connection for initial package installation
* Modern web browser
* Git

The project can be executed on Windows, Linux, or macOS with a compatible Python environment.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-doctor-visits-analytics.git
```

Move into the project directory:

```bash
cd healthcare-doctor-visits-analytics
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

The project dependencies are defined in `requirements.txt`.

Main libraries include:

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
openpyxl
```

---

# ▶️ Running the Project

## Run the Complete Analysis

From the project root directory:

```bash
python src/analysis.py
```

This performs:

* Data loading
* Data quality analysis
* Exploratory analysis
* Visualization generation
* Statistical/group analysis
* Model training
* Model evaluation
* Output generation

---

## Launch the Streamlit Dashboard

```bash
streamlit run src/app.py
```

After execution, Streamlit will provide a local URL that can be opened in a web browser.

---

# 📁 Generated Outputs

The analysis automatically generates several outputs.

## Visualizations

```text
outputs/
├── 01_visits_distribution.png
├── 02_age_distribution.png
├── 03_visits_by_gender.png
├── 04_reduced_vs_visits.png
├── 05_illness_vs_visits.png
└── 06_correlation_heatmap.png
```

## Data Analysis Results

```text
outputs/
├── data_quality.csv
├── descriptive_statistics.csv
└── group_analysis.txt
```

## Machine Learning Results

```text
outputs/
├── model_comparison.csv
└── test_predictions.csv
```

---

# 📑 Project Reports

The project includes professionally prepared reports in both Word and PDF formats.

### Word Report

```text
Healthcare_Doctor_Visits_Project_Report_with_Images.docx
```

### PDF Report

```text
Healthcare_Doctor_Visits_Project_Report_with_Images.pdf
```

The reports contain the project methodology, analysis, visualizations, machine-learning results, and conclusions.

---

# 🔁 Reproducibility

The project is designed to be reproducible.

The following configuration is used for the machine-learning experiment:

```text
Train-Test Split: 80 / 20
Random State: 42
Problem Type: Regression
```

The same dataset, preprocessing pipeline, model configuration, and random state can be used to reproduce the reported results.

---

# ⚠️ Limitations

Several limitations should be considered when interpreting the results.

1. The dataset represents a specific population and observation period.
2. The analysis is observational.
3. Statistical association does not establish causation.
4. Machine-learning performance depends on the available features and dataset characteristics.
5. The models should not be treated as medical diagnostic tools.
6. The results should not automatically be generalized to other populations.
7. Additional external healthcare variables may improve predictive performance.
8. More advanced validation and model-tuning approaches could provide a stronger evaluation.

---

# 🚀 Future Enhancements

The project can be extended in several directions.

### Advanced Machine Learning

* Gradient Boosting
* XGBoost
* LightGBM
* Support Vector Regression
* Hyperparameter optimization
* Cross-validation
* Ensemble modeling

### Explainable AI

* SHAP
* Feature importance
* Partial dependence analysis
* Model interpretation

### Advanced Analytics

* Statistical hypothesis testing
* Confidence intervals
* Feature engineering
* Outlier analysis
* Advanced segmentation

### Dashboard Enhancements

* Advanced KPI cards
* More interactive visualizations
* Model prediction interface
* Feature-importance visualization
* Automated reporting
* Cloud deployment

### Deployment

Potential deployment options include:

* Streamlit Community Cloud
* Docker
* Cloud-based hosting
* CI/CD using GitHub Actions

---

# 🎓 Academic & Learning Outcomes

This project demonstrates practical knowledge of:

* Data Analytics
* Data Science
* Healthcare Analytics
* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Data Visualization
* Feature Engineering
* Machine Learning
* Regression
* Model Evaluation
* Dashboard Development
* Data Interpretation
* Git and GitHub
* Reproducible Data Science

---

# 🔐 Data Privacy & Responsible Use

Healthcare-related datasets require careful handling.

This project is intended for **educational and analytical purposes**.

When working with real patient data:

* Personally identifiable information should be removed.
* Confidential healthcare information should not be publicly shared.
* Appropriate security controls should be implemented.
* Data should be handled according to applicable privacy and institutional requirements.
* Analytical results should not be presented as medical diagnoses.

---

# 📌 Important Interpretation Note

The purpose of this project is to identify patterns and associations in the supplied dataset.

The analysis should **not** be interpreted as:

* A medical diagnosis system
* A clinical decision-support system
* Evidence of causal relationships
* A replacement for professional medical judgment

The machine-learning models are experimental analytical models developed for educational purposes.

---

# 👨‍💻 Author

## Onkar Abhiman Londhe

**M.Tech Data Science | Data Analytics | Machine Learning**

### Technical Interests

* Data Analytics
* Data Science
* Machine Learning
* Artificial Intelligence
* Python
* SQL
* Power BI
* Excel
* Data Visualization
* Exploratory Data Analysis

---

# ⭐ Project Support

If this project helped you understand healthcare analytics, data science, or machine learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 📢 Share the project

---

# 📜 License

This project is intended primarily for educational, academic, and portfolio purposes.

If distributing the project under the MIT License, include a `LICENSE` file containing the appropriate MIT License text.

---

# ✅ Project Completion

| Component                 | Status      |
| ------------------------- | ----------- |
| Dataset Preparation       | ✅ Completed |
| Data Quality Analysis     | ✅ Completed |
| Exploratory Data Analysis | ✅ Completed |
| Statistical Analysis      | ✅ Completed |
| Data Visualization        | ✅ Completed |
| Feature Preprocessing     | ✅ Completed |
| Ridge Regression          | ✅ Completed |
| Random Forest Regression  | ✅ Completed |
| Model Evaluation          | ✅ Completed |
| Streamlit Dashboard       | ✅ Completed |
| Analysis Outputs          | ✅ Completed |
| Visualizations            | ✅ Completed |
| PDF Report                | ✅ Completed |
| Word Report               | ✅ Completed |
| GitHub Documentation      | ✅ Completed |

---

# 🏁 Conclusion

The **Healthcare Analytics for Doctor Visits** project provides an end-to-end demonstration of how healthcare-related data can be analyzed using modern data-science techniques.

The project begins with dataset inspection and preparation, followed by exploratory analysis, statistical comparisons, visualization, and machine-learning-based regression. An interactive Streamlit dashboard further enables users to explore the dataset through filters and visual summaries.

The analysis identifies meaningful associations between doctor visits and several health-related variables, particularly reduced activity and illness. Machine-learning models were also developed and evaluated to establish a baseline for predicting doctor visits.

Overall, the project demonstrates the complete workflow required to transform a raw healthcare dataset into **actionable analytical insights, predictive models, visualizations, and an interactive data application**.

---

<p align="center">

### 🏥 Healthcare Analytics • 📊 Data Science • 🤖 Machine Learning • 📈 Data Visualization

**Built with Python, Scikit-learn, Matplotlib & Streamlit**

</p>
