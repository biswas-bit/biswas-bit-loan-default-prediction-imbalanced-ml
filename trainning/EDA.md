
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
---

```python
cols_missing_10_60 = missing[(missing > 10) & (missing <= 60)].index.tolist()
print("Columns with 10–60% missing values:", cols_missing_10_60) 
```
## 3. Feature Types
 -Numerical features: continuous numeric columns excluding binary
 -Binary features: columns with exactly 2 unique values
 -Categorical features: object or category type

 ```python
numeric_cols = numerical_data(df)
binary_cols = binary_data(df)
cat_cols = categorical_data(df) 
```
**Observation:** Features separated to ensure proper preprocessing pipelines for modeling.
---

### 4. Skewness & Transformation
    -Skewness checked for numeric features:
     Skew > 1 → highly right-skewed
     Skew < -1 → highly left-skewed
-Transformation applied using **Yeo-Johnson Power Transformation**, which works for positive, zero, and negative values.

---
## 5.Skewness of Numerical Features
After applying Power Transformation to reduce skewness, the top skewed features are:

| Feature | Skewness |
|---------|----------|
| AMT_REQ_CREDIT_BUREAU_DAY  | 14.27 |
| AMT_REQ_CREDIT_BUREAU_HOUR | 13.64 |
| DAYS_EMPLOYED               | -1.44 |
| DEF_60_CNT_SOCIAL_CIRCLE    | 3.00 |
| NONLIVINGAREA_MODE          | 1.82 |

**Observation:**  
Count-based and heavily imbalanced features still exhibit high skew. This is **acceptable for tree-based models** like Random Forest and XGBoost, as they are robust to skewed distributions.
---
## 6.Correlation With Target
The correlation of numerical features with the target variable `TARGET` was computed:

```python
corr_with_target = df[numeric_cols + ['TARGET']].corr()['TARGET'].sort_values(ascending=False)
```
**Observations:**
  -Top correlated numeric features were identified.
  -No numeric feature had an absolute correlation greater than 0.5, indicating no obvious leakage.
  -Binary and categorical features were analyzed separately by examining default rates per category to ensure they have meaningful relationships with the target.
  ---
 ## 7. Data Leakage Check
To ensure that the dataset does not contain features that could leak information from the target, we checked for **target proxies** and **post-loan features**:

```python
leakage_features = ['DAYS_OVERDUE', 'LOAN_STATUS']
df.drop(columns=[f for f in leakage_features if f in df.columns], inplace=True)
```
---
## 8.Multicollinerity(corellation + VIF)
  **Corellation Heatmap**
       -Pairwise correlation among numerical features computed.
       -Upper triangle masked for readability.
       -Strong correlations (>|0.85|) noted, mainly among _AVG/_MEDI/_MODE variants.
```python 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()
```
**VIF(variance inflamation factor):**
   -Variance Inflation Factor calculated to check feature redundancy:
```python 
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_num = df[numeric_cols].dropna()
X_num = X_num.loc[:, X_num.std() > 0]
vif_df = pd.DataFrame()
vif_df["feature"] = X_num.columns
vif_df["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
vif_df.sort_values(by="VIF", ascending=False).head(10)
```
**Observation:** Features with VIF > 10 were noted but not removed, since tree-based models like Random Forest are robust to multicollinearity.
---





