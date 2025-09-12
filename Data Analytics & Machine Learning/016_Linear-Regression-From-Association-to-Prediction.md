## **Lecture Notes: Linear Regression - From Association to Prediction**

**Guide:** Vineet Tiwari
**Course:** Advanced Data Analysis and Statistical Inference
**Lecture Topic:** Modeling Relationships and Making Predictions with Simple Linear Regression

---

### **1. Introduction: Beyond Correlation**

Welcome, everyone. We have mastered the art of comparing groups (ANOVA) and quantifying associations (correlation). Now, we take a monumental step forward: **modeling**.

Correlation tells us *if* and *how strongly* two variables are related. **Linear Regression** allows us to describe *exactly how* they are related. It provides a mathematical equation that models a response variable `Y` (the outcome) as a function of a predictor variable `X` (the input). This empowers us to:
1.  **Understand Relationships:** Quantify the change in `Y` for a unit change in `X`.
2.  **Predict Outcomes:** Estimate the value of `Y` for a new, given value of `X`.
3.  **Quantify Uncertainty:** Provide confidence intervals for our predictions and for the relationship itself.

**A Critical Warning:** While incredibly powerful, a regression model shows **association, not causation.** Establishing causality requires controlled experiments, longitudinal data, or sophisticated causal inference techniques beyond the scope of this lecture. Always interpret results with domain knowledge and caution.

---

### **2. The Simple Linear Regression (SLR) Model**

The model for Simple Linear Regression is a formalization of drawing a "line of best fit" through a scatterplot.

**The Model Equation:**
$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

Let's break down each component:
*   **Y:** The **dependent variable** (response, outcome). This is what we are trying to predict or explain.
*   **X:** The **independent variable** (predictor, feature). This is what we use to make the prediction.
*   **β₀ (Intercept):** The expected value of `Y` when `X = 0`. *Warning:* This may not always have a sensible interpretation if `X=0` is outside the range of the observed data (extrapolation).
*   **β₁ (Slope):** The **key parameter of interest.** It represents the expected change in `Y` for a one-unit increase in `X`.
    *   If β₁ > 0, there is a positive relationship.
    *   If β₁ < 0, there is a negative relationship.
    *   If β₁ = 0, there is no linear relationship.
*   **ε (Error term):** This represents the random, unexplained variability. It accounts for all the reasons why the observed data points do not fall perfectly on a straight line. We assume ε ~ i.i.d. N(0, σ²).

**Estimation: Ordinary Least Squares (OLS)**
We don't know the true population parameters β₀ and β₁. We estimate them from our sample data using the **OLS** method. OLS finds the values for `b₀` and `b₁` (the estimated coefficients) that **minimize the Sum of Squared Residuals (SSR)**.

A **residual (e)** is the difference between the observed value (`y_i`) and the predicted value (`ŷ_i`) from the model: `e_i = y_i - ŷ_i`.

---

### **3. The Four Key Assumptions (LINE)**

For the OLS estimates to be the "Best Linear Unbiased Estimators" (BLUE) and for our inferences (p-values, CIs) to be valid, four key assumptions must hold. Remember the acronym **LINE**:

1.  **L - Linearity:** The relationship between `X` and `Y` must be linear. A scatterplot should show a roughly straight-line pattern. You cannot fit a straight line to a curved relationship.
2.  **I - Independence:** The observations, and therefore the residuals, must be independent of each other. This is often a data collection issue (e.g., no repeated measurements on the same subject unless accounted for).
3.  **N - Normality:** The residuals should be approximately normally distributed. This assumption is primarily important for generating accurate confidence intervals and hypothesis tests for the coefficients, especially with small sample sizes. OLS is robust to minor violations of this.
4.  **E - Equal Variance (Homoscedasticity):** The variance of the residuals should be constant across all levels of `X`. In a scatterplot of residuals vs. fitted values, the spread of points should be roughly the same across the x-axis. Heteroscedasticity (non-constant variance) invalidates standard errors.

**Diagnostic plots are essential for checking these assumptions.**

---

### **4. Hands-On Python Example: Full Workflow**

Let's walk through a complete regression analysis, from simulation to diagnostics.

```python
# SETUP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# For prettier plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. SIMULATE DATA THAT MEETS ASSUMPTIONS
np.random.seed(7) # For reproducibility
n = 150
X = np.linspace(0, 10, n) # Predictor variable

# True Model: Y = 2.5 + 1.8*X + ε, where ε ~ N(0, 2^2)
true_slope = 1.8
true_intercept = 2.5
noise = np.random.normal(loc=0, scale=2.0, size=n) # Normally distributed error
Y = true_intercept + true_slope * X + noise

# Create a DataFrame
df = pd.DataFrame({'x': X, 'y': Y})

# 2. EXPLORATORY DATA ANALYSIS: SCATTERPLOT
plt.figure()
sns.scatterplot(data=df, x='x', y='y', alpha=0.7)
plt.title('Scatterplot of Y vs X')
plt.xlabel('Predictor (X)')
plt.ylabel('Response (Y)')
plt.show()

# 3. FIT THE LINEAR MODEL
# Using the formula API: 'y ~ x' means "model y as a function of x"
model = smf.ols('y ~ x', data=df).fit()

# 4. INTERPRET THE MODEL SUMMARY
print("="*50)
print("REGRESSION MODEL SUMMARY")
print("="*50)
print(model.summary())
```

**Interpreting the Summary Output:**
The `model.summary()` provides a wealth of information. Key sections to focus on:
*   **Coefficients (coef):**
    *   `Intercept (b₀)`: The estimated value of Y when x=0.
    *   `x (b₁)`: The estimated slope. "For a one-unit increase in X, we expect Y to change by b₁ units."
*   **P>|t|:** The p-value for testing the null hypothesis that the corresponding coefficient is zero.
    *   For the slope (`x`), H₀: β₁ = 0. A small p-value (typically < 0.05) provides evidence that X is a statistically significant predictor of Y.
*   **[0.025 0.975]:** The 95% confidence interval for the coefficient. We are 95% confident that the true population slope (β₁) lies within this interval.
*   **R-squared:** The proportion of variance in Y that is explained by the linear model. Ranges from 0 to 1. An R² of 0.78 means 78% of the variation in Y is explained by X.

```python
# 5. VISUALIZE THE REGRESSION FIT AND UNCERTAINTY
# Create a grid of x values for prediction
x_grid = pd.DataFrame({'x': np.linspace(df['x'].min(), df['x'].max(), 200)})

# Get predictions for the grid: mean, CI for the mean, PI for a new observation
predictions = model.get_prediction(x_grid)
pred_frame = predictions.summary_frame(alpha=0.05) # 95% level

# Plot
plt.figure()
# 1. Scatterplot of raw data
sns.scatterplot(data=df, x='x', y='y', alpha=0.6, label='Observed Data')
# 2. Plot the fitted regression line
plt.plot(x_grid['x'], pred_frame['mean'], color='red', linewidth=2, label='Fitted Line')
# 3. Shade the 95% Confidence Interval (for the mean response)
plt.fill_between(x_grid['x'], pred_frame['mean_ci_lower'], pred_frame['mean_ci_upper'],
                 color='red', alpha=0.2, label='95% CI (Mean)')
# 4. Shade the 95% Prediction Interval (for a new observation)
plt.fill_between(x_grid['x'], pred_frame['obs_ci_lower'], pred_frame['obs_ci_upper'],
                 color='grey', alpha=0.1, label='95% PI (New Obs)')

plt.title('Linear Regression Fit with Confidence and Prediction Intervals')
plt.xlabel('Predictor (X)')
plt.ylabel('Response (Y)')
plt.legend()
plt.show()

# 6. CHECK MODEL ASSUMPTIONS WITH DIAGNOSTIC PLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
model_fittedvals = model.fittedvalues
model_residuals = model.resid

# Plot 1: Residuals vs Fitted - Check for Linearity & Homoscedasticity
axes[0, 0].scatter(model_fittedvals, model_residuals, alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted Values')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
# A good plot shows no pattern (random cloud). A funnel shape indicates heteroscedasticity.

# Plot 2: Q-Q Plot - Check for Normality of Residuals
sm.qqplot(model_residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot of Residuals')
# Points should closely follow the diagonal line.

# Plot 3: Scale-Location Plot - Another check for Homoscedasticity
abs_sqrt_resid = np.sqrt(np.abs(model_residuals))
axes[1, 0].scatter(model_fittedvals, abs_sqrt_resid, alpha=0.7)
axes[1, 0].set_title('Scale-Location Plot')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
# A horizontal band indicates constant variance.

# Plot 4: Residuals vs Leverage - Check for influential points
sm.graphics.influence_plot(model, ax=axes[1, 1], criterion="cooks")
axes[1, 1].set_title('Residuals vs Leverage')
# Points in the top right or top left corner may be highly influential.

plt.tight_layout()
plt.show()

# 7. (BONUS) MAKE A PREDICTION FOR A NEW DATA POINT
new_x = pd.DataFrame({'x': [2, 5, 8]}) # New values to predict for
new_pred = model.get_prediction(new_x)
prediction_summary = new_pred.summary_frame(alpha=0.05)

print("\nPrediction for new X values:")
print(prediction_summary.round(2))
```

---

### **5. Inference and Uncertainty: Two Types of Intervals**

It is crucial to distinguish between two different intervals:
*   **Confidence Interval (CI) for the Mean Response:** This interval represents the uncertainty in estimating the *average* value of `Y` for a given `X`. It is narrower and answers the question: "What is the range for the *mean* exam score for students who study 5 hours?"
*   **Prediction Interval (PI) for a New Observation:** This interval represents the uncertainty in predicting an *individual* new value of `Y` for a given `X`. It is much wider because it must account for both the uncertainty in the mean estimate *and* the inherent randomness (ε) of the data point itself. It answers the question: "What is the range for the exam score of a *specific, new* student who studies 5 hours?"

---

### **6. Common Pitfalls and Remedies**

*   **Non-linearity:** If the scatterplot or residual plot shows a curve, don't use a straight line! **Remedy:** Transform variables (e.g., log(X), √X) or use polynomial regression.
*   **Heteroscedasticity:** If the residual plot shows a funnel shape, your standard errors are invalid. **Remedy:** Use robust standard errors (e.g., `cov_type='HC3'` in `model.fit(cov_type='HC3')`) or transform the Y variable.
*   **Outliers and Influential Points:** Points that have high leverage or large residuals can distort the regression line. **Remedy:** Investigate these points for data errors. If valid, consider robust regression techniques.
*   **Extrapolation:** Predicting for X values outside the range of your training data is dangerous and often highly inaccurate. The model's behavior in unobserved regions is unknown. **Don't do it.**

---

### **7. Reporting Results**

"A simple linear regression was performed to quantify the relationship between [X] and [Y]. The model explained a significant proportion of variance (R² = .XX, F(1, XX) = XX.XX, p < .001). A one-unit increase in [X] was associated with a [b₁] unit increase in [Y] (95% CI [LL, UL], p < .001). Diagnostic plots of the residuals indicated no severe violations of the assumptions of linearity, normality, or homoscedasticity."

---

### **8. Key Takeaways**

1.  **Regression models relationships** for both explanation and prediction.
2.  **Always check the assumptions (LINE)** before trusting the model's inferences. Your diagnostics are just as important as your results.
3.  **Distinguish between CIs and PIs.** One is for the mean, the other is for an individual prediction.
4.  **Correlation ≠ Causation.** A significant slope does not mean X causes Y.
5.  **Beware of extrapolation** and influential outliers.

**Next Lecture:** We will extend these concepts to **Multiple Linear Regression**, where we can model a response variable using several predictors simultaneously, which is the true workhorse of modern data analysis.

**Are there any questions?**