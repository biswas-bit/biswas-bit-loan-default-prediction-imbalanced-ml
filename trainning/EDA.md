@'
# Exploratory Data Analysis (EDA) – Loan Default Risk Detection

## 1. Dataset Overview

- Total rows: 610 
- Total columns: 122
- Target variable: TARGET (0 = non-default, 1 = default)
- Data types: numeric, binary, categorical

**Key Insight:** Dataset contains a mix of numeric, categorical, and binary features. Target is binary, and the dataset is suitable for supervised classification.

---

## 2. Missing Values Treatment

- Columns with > 60% missing values: removed  
- Columns with 10–60% missing values: replaced with median (numerical) or 'unknown' (categorical)  
- Columns with < 10% missing values: replaced with median/mode  

```python
cols_missing_10_60 = missing[(missing > 10) & (missing <= 60)].index.tolist()
print("Columns with 10–60% missing values:", cols_missing_10_60)