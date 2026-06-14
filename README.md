# Heart Disease Data Analysis & Preprocessing

This project demonstrates the complete Machine Learning data preparation workflow using the Heart Disease dataset. The notebook focuses on data cleaning, exploratory data analysis (EDA), feature engineering, encoding categorical variables, and feature scaling before model training.

## Project Overview

The objective of this project is to analyze heart disease-related patient data and prepare it for machine learning models.

The workflow includes:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Handling Missing/Invalid Values
- Duplicate Detection
- Feature Encoding
- Feature Scaling
- Correlation Analysis

## Dataset

The dataset contains patient health information such as:

- Age
- Sex
- Resting Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Fasting Blood Sugar
- Oldpeak
- HeartDisease (Target Variable)

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## Steps Performed

### 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("heart.csv")
```

### 2. Data Inspection

- Shape of dataset
- Data types
- Statistical summary
- Duplicate detection
- Null value detection

### 3. Data Cleaning

Invalid values were identified and replaced:

- Cholesterol values equal to 0 were replaced with the mean cholesterol value.
- RestingBP values equal to 0 were replaced with the mean RestingBP value.

### 4. Exploratory Data Analysis (EDA)

Visualizations performed:

- Histograms
- Count Plots
- Box Plots
- Violin Plots
- Correlation Heatmap

### 5. Categorical Feature Encoding

One-Hot Encoding was applied using:

```python
pd.get_dummies(drop_first=True)
```

This helps avoid the Dummy Variable Trap.

### 6. Feature Scaling

Standardization was performed on numerical features:

```python
StandardScaler()
```

Scaled Features:

- Age
- RestingBP
- Cholesterol
- MaxHR
- Oldpeak

## Project Structure

```
├── ML_basics_heart.ipynb
├── heart.csv
└── README.md
```

## Key Learnings

- Data Cleaning Techniques
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Standard Scaling
- Correlation Analysis

## Future Improvements

- Train Machine Learning Models
- Compare Classification Algorithms
- Hyperparameter Tuning
- Model Evaluation Metrics
- Deployment using Streamlit or Flask

## Author

Ayush Gupta

