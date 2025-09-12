## **Detailed Lecture Notes: Logistic Regression - Modeling Binary Outcomes**

**Guide:** Vineet Tiwari
**Course:** Advanced Data Analysis and Statistical Inference
**Lecture Topic:** Classification with Logistic Regression, Interpretation, and Evaluation

---

### **1. Introduction: The Need for a New Model When Y is Binary**

Our journey with Multiple Linear Regression equipped us to model continuous outcomes. We now confront a fundamentally different type of problem: **classification**, where the response variable \(Y\) is **categorical**. In this lecture, we focus on the most common case: **binary outcomes** (e.g., 0/1, Yes/No, Success/Failure, Customer Churns/Stays).

Why can't we use linear regression for a binary outcome?
1.  **Invalid Predictions:** Linear regression can produce predicted values outside the [0, 1] probability range, which are nonsensical.
2.  **Invalid Assumptions:** The error terms are not normally distributed for a binary Y; they follow a Bernoulli distribution.
3.  **Non-constant Variance:** The variance of the error term depends on the value of X, violating the homoscedasticity assumption.

**Logistic Regression** solves these problems. It does not model the outcome \(Y\) directly. Instead, it models the *probability* that \(Y\) belongs to a particular category (conventionally, \(Y=1\)), ensuring all predictions are bounded between 0 and 1.

---

### **2. The Logistic Model: From Linear Predictors to Probabilities**

The core of logistic regression is the **logistic function** (or **sigmoid function**), which maps any real number to a value between 0 and 1.

**The Probability Model:**
$$
\Pr(Y=1 \mid X) = p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}
$$

This elegant transformation takes our familiar linear combination of predictors \(\beta_0 + \sum \beta_j X_j\) (called the **linear predictor**) and squeezes it into the (0, 1) interval.

#### **Interpretation via Odds and Log-Odds**

While the probability \(p(X)\) is the direct output, the coefficients \(\beta_j\) are not interpreted as a linear change in probability. The model is linear in the **log-odds** (logit).

**The Log-Odds (Logit) Form:**
$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
$$
Where \(\frac{p(X)}{1 - p(X)}\) is the **odds** of the event occurring (e.g., the odds of a customer churning).

**Interpreting Coefficients:**
*   **\(\beta_j\):** A one-unit increase in \(X_j\) is associated with a \(\beta_j\) change in the *log-odds* of \(Y=1\), holding all other predictors constant.
*   **Odds Ratio (\(e^{\beta_j}\)):** This is often more intuitive. A one-unit increase in \(X_j\) is associated with a **multiplicative change of \(e^{\beta_j}\) in the odds** of the event.
    *   \(e^{\beta_j} = 1\): No effect.
    *   \(e^{\beta_j} > 1\): The odds increase.
    *   \(e^{\beta_j} < 1\): The odds decrease.

*Example:* If \(\beta_{\text{income}} = 0.5\), then \(e^{0.5} \approx 1.65\). We would interpret this as: "For each additional unit of income, the odds of the event occurring are multiplied by 1.65 (i.e., they increase by 65%), holding other variables constant."

---

### **3. Assumptions and Considerations**

Logistic regression has a different set of assumptions than linear regression:
*   **Binary Response:** The dependent variable must be binary.
*   **Independence of Observations:** Data points must not be correlated (e.g., no repeated measures).
*   **Linearity in Log-Odds:** The relationship between the predictors and the *log-odds* of the outcome is linear. This can be checked with the **Box-Tidwell test** (plotting predictors against log-odds) and may require adding polynomial or spline terms for continuous predictors.
*   **No Severe Multicollinearity:** As in MLR, high correlation between predictors inflates standard errors. Check using Variance Inflation Factor (VIF).
*   **Large Sample Size / Sufficient Events:** A common rule of thumb is at least 10-20 events (Y=1) *per predictor variable* in the model to avoid overfitting and ensure stable estimates.

---

### **4. Worked Example in Python: A Complete Workflow**

The following code simulates data, fits a model, and performs a comprehensive evaluation.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(21)

# Simulate a realistic dataset with a non-linear effect
n = 1200
x1 = np.random.normal(0, 1.0, n)  # e.g., standardized account age
x2 = np.random.normal(0, 1.0, n)  # e.g., standardized spending
# Create a linear combination in log-odds space, including a quadratic term for x1
log_odds = -0.5 + 1.2*x1 + 0.8*x2 - 0.6*(x1**2)
# Transform log-odds to probability using the logistic function
p = 1 / (1 + np.exp(-log_odds))
# Generate binary outcomes (0/1) from these probabilities
y = np.random.binomial(1, p, n)

df = pd.DataFrame({'Account_Age': x1, 'Spending': x2, 'Churned': y})
print(df['Churned'].value_counts(normalize=True)) # Check class balance

# Split the data into training and test sets
# 'stratify=y' ensures the proportion of 1s/0s is the same in both sets
X = df[['Account_Age','Spending']]
y = df['Churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Standardize features (Crucial for regularized models to ensure penalties are applied fairly)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Fit a logistic regression model with L2 regularization (the default)
# 'max_iter' may need to be increased for large datasets or complex models
clf = LogisticRegression(penalty='l2', max_iter=200, solver='lbfgs', random_state=42)
clf.fit(X_train_s, y_train)

# Examine the model coefficients (in log-odds)
print('Intercept (log-odds):', clf.intercept_)
print('Coefficients (log-odds):', clf.coef_)
# Convert coefficients to Odds Ratios for interpretation
print('Odds Ratios:', np.exp(clf.coef_))

# Predict probabilities for the positive class (class 1)
y_pred_proba = clf.predict_proba(X_test_s)[:, 1]
# Predict class labels using the default 0.5 threshold
y_pred = clf.predict(X_test_s)

# --- Comprehensive Model Evaluation ---

# 1. Confusion Matrix: The cornerstone of classification evaluation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()

# 2. Classification Report: Precision, Recall, F1-Score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

# 3. ROC Curve and AUC: Overall performance across all thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 4. Precision-Recall Curve: Especially important for imbalanced datasets
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {average_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()
```

**Code Walkthrough and Teaching Notes:**
*   **Data Simulation:** We create a scenario where the log-odds of churning depend on `Account_Age` in a non-linear way (a quadratic term). Real-world probabilities are rarely purely linear.
*   **Train-Test Split:** The `stratify` parameter is critical for maintaining class balance in both sets.
*   **Standardization:** Essential for regularized models. Even without regularization, it helps with coefficient interpretation if predictors are on different scales.
*   **Interpretation:** We print both the raw coefficients (log-odds change) and the exponentiated coefficients (odds ratios). The odds ratios are often reported to stakeholders.
*   **Evaluation:** We move beyond simple accuracy.
    *   The **Confusion Matrix** breaks down error types (False Positives vs. False Negatives).
    *   The **Classification Report** provides precision (how many predicted positives are real), recall (how many real positives we found), and their harmonic mean, the F1-score.
    *   The **ROC Curve** shows the trade-off between True Positive Rate and False Positive Rate at all thresholds. **AUC (Area Under Curve)** summarizes overall performance (0.5 = random, 1.0 = perfect).
    *   The **Precision-Recall Curve** is often more informative than ROC for imbalanced data, as it focuses on the performance regarding the positive class.

---

### **5. The Critical Issue of Class Imbalance and Threshold Tuning**

The default **decision threshold** is 0.5. This is often not optimal, especially with imbalanced classes (e.g., 1% churn rate).

*   **Why 0.5 might be bad:** If it's 10x more costly to miss a positive (False Negative) than to falsely flag a negative (False Positive), a threshold of 0.3 might be better, capturing more true positives at the cost of more false alarms.
*   **Solutions:**
    1.  **Use `class_weight='balanced'`** in `LogisticRegression()`. This tells the algorithm to penalize errors on the minority class more heavily.
    2.  **Resample the data** (e.g., oversample the minority class with SMOTE or undersample the majority class).
    3.  **Tune the threshold** using the validation set. Use the Precision-Recall curve to find a threshold that achieves your desired trade-off.

---

### **6. Communicating Results and Pitfalls**

*   **Reporting:** Report **Odds Ratios** with their **95% Confidence Intervals** (`np.exp(coefficient ± 1.96 * standard_error)`). This communicates both the effect size and its precision.
*   **Pitfalls:**
    *   **Perfect Separation:** If a predictor perfectly separates the classes, the coefficient will blow up to infinity. **Solution:** Use regularization (L1/L2) or collect more data.
    *   **Overfitting:** Always validate on a hold-out test set. Simpler models with fewer features are often more robust.

---

### **7. Key Takeaways**

1.  **Purpose:** Logistic regression is the go-to method for modeling the probability of a binary outcome. It outputs calibrated probabilities between 0 and 1.
2.  **Interpretation:** Coefficients represent changes in **log-odds**. Exponentiating them yields **Odds Ratios**, which are multiplicative effects on the odds of the event.
3.  **Evaluation:** Never rely on accuracy alone. Use a suite of tools: the **Confusion Matrix**, **ROC/AUC**, and **Precision-Recall Curves**.
4.  **Imbalance:** Be acutely aware of class imbalance. Address it through class weighting, resampling, and careful **threshold tuning** based on business costs.

---

### **9. Next Lecture Preview**

We will now step into the world of non-linear, tree-based models.

**Next Lecture: Decision Trees for Classification & Regression**

*   **Concept:** Learn how simple, hierarchical "if-else" rules can be used to make predictions.
*   **Splitting Criteria:** Understand how trees decide where to split using **Gini Impurity** and **Information Gain**.
*   **Strengths & Weaknesses:** Discuss the interpretability of trees vs. their tendency to **overfit**.
*   **Comparison:** Contrast the "white-box" nature of a single tree with the logistic model we built today.

**Are there any questions on Logistic Regression before we move on?**