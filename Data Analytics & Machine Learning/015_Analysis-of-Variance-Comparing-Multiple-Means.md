## **Lecture Notes: Analysis of Variance (ANOVA) - Comparing Multiple Means**

**Guide:** Vineet Tiwari

**Course:** Advanced Data Analysis and Statistical Inference

**Lecture Topic:** Testing Mean Differences Across Several Groups with ANOVA

---

### **1. Introduction: The Problem with Multiple T-Tests**

Welcome, everyone. Thus far, our toolkit for comparing means has been the **t-test**—a powerful method for determining if there is a statistically significant difference between the means of *two* groups.

But what happens when our research question involves **three or more groups**?
*   Comparing the effectiveness of three different teaching methods.
*   Testing the yield of four different fertilizers.
*   Analyzing customer satisfaction across five regional branches.

A naive approach would be to perform multiple pairwise t-tests (e.g., Method A vs. B, A vs. C, B vs. C). However, this leads to the **multiple comparisons problem**. With each test performed at a significance level of α = 0.05, the overall chance of committing at least one **Type I error** (a false positive) increases dramatically. For 3 groups (3 tests), the family-wise error rate rises to about 14%. For 5 groups (10 tests), it's about 40%!

**Analysis of Variance (ANOVA)** solves this problem. It is an **"omnibus" test** that allows us to compare multiple group means simultaneously with a single test, controlling the overall Type I error rate at the chosen α level.

---

### **2. The Core Intuition: Partitioning Variance**

ANOVA works by comparing two different sources of variability in the data:

1.  **Between-Group Variability:** How much the group *means* differ from the overall (grand) mean. If our treatments are effective, we expect this variability to be large.
2.  **Within-Group Variability:** How much the individual *observations* within each group differ from their own group mean. This represents natural, random variation that is always present.

**The Logic of the F-test:**
> If the **between-group variability** is significantly larger than the **within-group variability**, then the differences between the group means are unlikely to be due to random chance alone. We can conclude that at least one group mean is different.

---

### **3. The Formal Setup: Hypotheses and the F-Statistic**

#### **A. Hypotheses**
*   **Null Hypothesis (H₀):** μ₁ = μ₂ = ... = μₖ
    *   *All* population means are equal.
*   **Alternative Hypothesis (H₁):** At least one population mean is different.
    *   It is important to note that ANOVA does not tell you *which* mean is different or *how many* are different, only that at least one is.

#### **B. The F-Statistic**
The test statistic for ANOVA is the **F-statistic**, which is a ratio of the two sources of variance:
$$
F = \frac{\text{Mean Square Between (MS}_{\text{between}})}{\text{Mean Square Within (MS}_{\text{within}})} = \frac{\text{Variance between groups}}{\text{Variance within groups}}
$$
*   **Mean Square Between (MS_between):** The between-group variability, calculated as the Sum of Squares Between (SS_between) divided by its degrees of freedom (df_between = k - 1, where k is the number of groups).
*   **Mean Square Within (MS_within):** The within-group variability (often called "error"), calculated as the Sum of Squares Within (SS_within) divided by its degrees of freedom (df_within = N - k, where N is the total sample size).

**Interpretation:**
*   **F ≈ 1:** Suggests the null hypothesis is true. The variance between groups is about the same as the variance within groups.
*   **F >> 1:** Suggests the null hypothesis is false. The variance between groups is large compared to the variance within groups.

We compare the calculated F-statistic to a critical value from the F-distribution (with df_between and df_within) to determine the p-value.

---

### **4. Assumptions of One-Way ANOVA**

For the results of an ANOVA to be valid, three key assumptions must be reasonably met:

1.  **Independence:** Observations must be independent of each other, both within and between groups. This is a design issue (e.g., random sampling/assignment).
2.  **Normality:** The residuals (the differences between observations and their group mean) should be approximately normally distributed for each group. ANOVA is robust to minor violations of this assumption, especially with larger sample sizes.
3.  **Homogeneity of Variances (Homoscedasticity):** The population variance within each group should be approximately equal. This is a crucial assumption.

---

### **5. Hands-On Python Example**

Let's walk through a complete one-way ANOVA, from data simulation to interpretation.

```python
# SETUP
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# 1. SIMULATE DATA
np.random.seed(42)  # For reproducibility
n = 40  # Sample size per group

# Simulate test scores for three different teaching methods
# Method A: Mean = 75, SD = 8
# Method B: Mean = 79, SD = 8
# Method C: Mean = 85, SD = 8
method_A = np.random.normal(loc=75, scale=8, size=n)
method_B = np.random.normal(loc=79, scale=8, size=n)
method_C = np.random.normal(loc=85, scale=8, size=n)

# Create a DataFrame for easier analysis
df = pd.DataFrame({
    'score': np.concatenate([method_A, method_B, method_C]),
    'method': (['A'] * n) + (['B'] * n) + (['C'] * n)
})

# 2. EXPLORATORY DATA ANALYSIS (VISUALIZATION)
plt.figure(figsize=(10, 6))

# Boxplot to see group distributions, medians, and variability
plt.subplot(1, 2, 1)
df.boxplot('score', by='method', ax=plt.gca())
plt.title('Scores by Teaching Method')
plt.suptitle('')  # Remove pandas auto-title
plt.ylabel('Test Score')

# Plot group means with error bars (95% CI for the mean)
plt.subplot(1, 2, 2)
group_means = df.groupby('method')['score'].mean()
group_std = df.groupby('method')['score'].std()
group_n = df.groupby('method')['score'].count()
group_ci = 1.96 * (group_std / np.sqrt(group_n)) # Approx 95% CI

plt.errorbar(x=group_means.index, y=group_means, yerr=group_ci,
             fmt='o', capsize=5, label='Mean ± 95% CI')
plt.xlabel('Teaching Method')
plt.ylabel('Mean Test Score')
plt.title('Mean Scores with Confidence Intervals')
plt.legend()

plt.tight_layout()
plt.show()

# 3. CHECK ASSUMPTIONS

# Check Normality of Residuals (QQ-Plot is best)
model = smf.ols('score ~ C(method)', data=df).fit()
residuals = model.resid
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Check Homogeneity of Variances (Levene's Test)
# H₀: All group variances are equal.
stat, p_val_levene = stats.levene(df[df['method']=='A']['score'],
                                  df[df['method']=='B']['score'],
                                  df[df['method']=='C']['score'])
print(f"Levene's Test for Homogeneity of Variances: W={stat:.3f}, p={p_val_levene:.4f}")
if p_val_levene > 0.05:
    print("--> Assumption of equal variances is not violated.\n")
else:
    print("--> Warning: Variances may not be equal. Consider a Welch's ANOVA.\n")

# 4. PERFORM ONE-WAY ANOVA

# Method 1: Using SciPy's f_oneway (simple, but less output)
F_stat, p_val = stats.f_oneway(method_A, method_B, method_C)
print("=== SCIPY ONE-WAY ANOVA ===")
print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {p_val:.6f}\n")

# Method 2: Using statsmodels (recommended - provides full ANOVA table)
model = smf.ols('score ~ C(method)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2) # Type 2 ANOVA
print("=== STATSMODELS ANOVA TABLE ===")
print(anova_table)
print("\n")

# 5. POST-HOC TEST (IF ANOVA IS SIGNIFICANT)
if p_val < 0.05:
    print("Since p < 0.05, we reject the null hypothesis.")
    print("Performing Tukey's HSD post-hoc test to find which groups differ...\n")

    # Tukey's Honest Significant Difference test
    tukey = pairwise_tukeyhsd(endog=df['score'],   # Data
                              groups=df['method'], # Group labels
                              alpha=0.05)          # Significance level

    print(tukey.summary())
    # tukey.plot_simultaneous() # Uncomment to see a visual plot of comparisons
else:
    print("P-value > 0.05. Fail to reject H₀. No post-hoc test needed.")

# 6. CALCULATE EFFECT SIZE (PRACTICAL SIGNIFICANCE)
# Eta-squared (η²) = SS_between / SS_total
ss_between = anova_table['sum_sq']['C(method)']
ss_total = anova_table['sum_sq']['C(method)'] + anova_table['sum_sq']['Residual']
eta_squared = ss_between / ss_total

print(f"\nEffect Size (Eta-squared, η²): {eta_squared:.3f}")
# Interpretation: ~0.01 (small), ~0.06 (medium), ~0.14 (large)
```

**Expected Output & Analysis:**
The code will generate visualizations showing the data distribution for each group. The Levene's test p-value will likely be > 0.05, confirming the equal variance assumption is met. The Q-Q plot will show points roughly on the line, suggesting normality.

The ANOVA table will show a significant F-statistic (e.g., F(2, 117) = ~10, p < .001). This tells us to reject the null hypothesis: at least one teaching method leads to a different mean score.

The **Tukey HSD test** will then reveal the specific pairwise differences. The output will show:
*   A significant difference between Method C and Method A (p < 0.05).
*   A significant difference between Method C and Method B (p < 0.05).
*   A non-significant difference between Method A and Method B (p > 0.05).

Finally, the effect size (η²) will quantify the magnitude of the difference, telling us how much of the total variance in scores is explained by the teaching method.

---

### **6. What if Assumptions are Violated?**

*   **Failed Normality?** Consider a **non-parametric alternative: the Kruskal-Wallis H-test** (`scipy.stats.kruskal`). It's the non-parametric version of one-way ANOVA.
*   **Failed Homogeneity of Variances?** Use **Welch's ANOVA** (`pingouin.anova` or `scipy.stats.f_oneway` with `equal_var=False`), which does not assume equal variances.

---

### **7. Reporting Results in a Paper**

"A one-way ANOVA was conducted to compare the effect of three teaching methods (A, B, C) on test scores. The assumptions of independence and homogeneity of variances were met, as assessed by Levene's test (p = .XX). Residuals were approximately normally distributed. The ANOVA revealed a statistically significant effect of teaching method on scores, F(2, 117) = [F-value], p < .001, η² = .14, a large effect size.

Post-hoc analyses using Tukey's HSD test indicated that the mean score for Method C (M = XX, SD = XX) was significantly higher than both Method B (M = XX, SD = XX, p = .012) and Method A (M = XX, SD = XX, p < .001). There was no significant difference between Methods A and B (p = .XX)."

---

### **8. Key Takeaways**

1.  **ANOVA is for 3+ Groups:** It's an omnibus test that controls the Type I error rate when comparing multiple means.
2.  **It Compares Variances:** The F-test ratio determines if between-group differences are larger than expected by random chance.
3.  **Check Assumptions:** Always verify independence, normality of residuals, and homogeneity of variances before interpreting results.
4.  **A Significant F is Just the Start:** A significant ANOVA only tells you that not all means are equal. You **must** follow up with a **post-hoc test** (like Tukey's HSD) to determine which specific groups differ.
5.  **Report Effect Sizes:** Always report a measure of practical significance (like η²) alongside the p-value.

**Next Lecture:** We will transition from comparing groups to modeling relationships between continuous variables with **Linear Regression.**

**Are there any questions?**