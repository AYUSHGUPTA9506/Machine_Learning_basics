# 🤖 Machine Learning Basics

A hands-on collection of end-to-end machine learning projects covering real-world datasets — from raw data exploration to model-ready feature sets.

---

## 📁 Repository Structure

```
Machine_Learning_basics/
│
├── ML_basics_insurance.ipynb   # Insurance charges prediction (regression)
├── insurance.csv               # Medical insurance dataset (1,338 records)
│
├── ML_basics_heart.ipynb       # Heart disease prediction (classification)
├── heart.csv                   # Heart disease dataset
│
└── README.md
```

---

## 📓 Projects

### 1. 💰 Insurance Charges Prediction

**Notebook:** `ML_basics_insurance.ipynb`  
**Dataset:** `insurance.csv`

Predicts individual medical insurance charges using patient attributes like age, BMI, smoking status, and region.

#### Dataset Overview

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age of the policyholder |
| `sex` | categorical | Gender (male / female) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | categorical | Smoking status (yes / no) |
| `region` | categorical | US region (northeast / northwest / southeast / southwest) |
| `charges` | float | Medical insurance cost billed (**target variable**) |

**Records:** 1,338 rows × 7 columns

#### Pipeline Walkthrough

**Exploratory Data Analysis (EDA)**
- Distribution plots (histplot with KDE) for `age`, `bmi`, `children`, `charges`
- Count plots for categorical features: `sex`, `smoker`, `children`
- Box plots for outlier detection
- Correlation heatmap (Pearson) across numeric features

**Data Cleaning & Preprocessing**
- Removed duplicate rows
- Label encoded binary categories: `sex` → `is_female`, `smoker` → `is_smoker`
- One-hot encoded `region` using `pd.get_dummies` (drop_first=True)

**Feature Engineering**
- Created `bmi_category` using `pd.cut` with clinically standard bins:
  - Underweight (< 18.5) · Normal (18.5–24.9) · Overweight (25–29.9) · Obese (≥ 30)
- One-hot encoded `bmi_category` (drop_first=True)

**Feature Scaling**
- Applied `StandardScaler` to continuous features: `age`, `bmi`, `children`

**Feature Selection**
- Pearson correlation for numeric features vs `charges`
- Chi-square test (`chi2_contingency`) for categorical features vs binned charges
- Final selected features: `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_obese`

---

### 2. ❤️ Heart Disease Prediction

**Notebook:** `ML_basics_heart.ipynb`  
**Dataset:** `heart.csv`

Covers preprocessing and exploratory analysis for a heart disease classification task.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots (histplot, boxplot, heatmap, countplot) |
| `scikit-learn` | StandardScaler, ML model preparation |
| `scipy` | Pearson correlation, Chi-square feature selection |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AYUSHGUPTA9506/<repo-name>.git
cd Machine_Learning_basics
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Run the notebook

Open in Jupyter or VS Code:

```bash
jupyter notebook ML_basics_insurance.ipynb
```

> **Note:** The notebook reads `insurance.csv` from the same folder. Make sure both files are in the same directory before running.

---

## 📊 Key Insights

- **Smoking** is the strongest predictor of insurance charges — smokers are charged significantly more
- **BMI ≥ 30 (Obese)** is a key engineered feature that correlates strongly with high charges
- **Age** shows a consistent positive correlation with charges
- **Region** has relatively low predictive power compared to lifestyle factors

---

## 👤 Author

**Ayush Gupta**  
Data Science & AI | ML · Deep Learning · Generative AI · MLOps  
[GitHub](https://github.com/AYUSHGUPTA9506)
