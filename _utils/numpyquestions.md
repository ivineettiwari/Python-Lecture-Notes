* **NumPy Core + Matrix**
* **Statistics Basics to Advanced**
* **Probability**
* **Z / T / Hypothesis / CLT / P-value**
* **Inferential Statistics**
---

# ✅ SECTION A — SHAPE AND STATS WITH NUMPY (Q1–Q10)

---

### Q1. What is NumPy?

**Answer:**
NumPy is Python’s numerical computing library used for fast multidimensional array operations.

---

### Q2. How do you import NumPy?

```python
import numpy as np
```

---

### Q3. Create a 1D NumPy array of numbers 1 to 5.

```python
a = np.array([1,2,3,4,5])
```

---

### Q4. How to check shape of array?

```python
a.shape
```

**Answer:** returns tuple of dimensions.

---

### Q5. Difference between `shape` and `size`?

**Answer:**

* `shape` → dimensions
* `size` → total elements

---

### Q6. How to find mean of array?

```python
np.mean(a)
```

---

### Q7. How to find median?

```python
np.median(a)
```

---

### Q8. How to find standard deviation?

```python
np.std(a)
```

---

### Q9. How to find variance?

```python
np.var(a)
```

---

### Q10. How to find max-min range?

```python
np.ptp(a)
```

---

# ✅ SECTION B — ARRAY MAGIC: INDEXING AND SLICING (Q11–Q18)

---

### Q11. Access 3rd element.

```python
a[2]
```

---

### Q12. Last element?

```python
a[-1]
```

---

### Q13. Slice first 4 values.

```python
a[:4]
```

---

### Q14. Reverse array.

```python
a[::-1]
```

---

### Q15. Boolean indexing example?

```python
a[a>3]
```

---

### Q16. Fancy indexing?

```python
a[[0,2,4]]
```

---

### Q17. Replace second element by 100.

```python
a[1]=100
```

---

### Q18. Select even values.

```python
a[a%2==0]
```

---

# ✅ SECTION C — CRAFTING ARRAYS: FROM IMPORT TO STATS (Q19–Q28)

---

### Q19. Create zeros array of size 5.

```python
np.zeros(5)
```

---

### Q20. Create ones matrix 3x3.

```python
np.ones((3,3))
```

---

### Q21. Create identity matrix.

```python
np.eye(3)
```

---

### Q22. Create range 1 to 10.

```python
np.arange(1,11)
```

---

### Q23. Create evenly spaced values.

```python
np.linspace(0,1,5)
```

---

### Q24. Reshape 1D to 2x3.

```python
a.reshape(2,3)
```

---

### Q25. Flatten matrix.

```python
a.flatten()
```

---

### Q26. Concatenate arrays.

```python
np.concatenate((a,b))
```

---

### Q27. Vertical stack?

```python
np.vstack((a,b))
```

---

### Q28. Horizontal stack?

```python
np.hstack((a,b))
```

---

# ✅ SECTION D — THE MATRIX TOOLKIT (Q29–Q36)

---

### Q29. Matrix addition?

```python
A+B
```

---

### Q30. Matrix subtraction?

```python
A-B
```

---

### Q31. Elementwise multiplication?

```python
A*B
```

---

### Q32. Matrix multiplication?

```python
np.dot(A,B)
```

---

### Q33. Transpose of matrix?

```python
A.T
```

---

### Q34. Determinant?

```python
np.linalg.det(A)
```

---

### Q35. Inverse?

```python
np.linalg.inv(A)
```

---

### Q36. Eigen values?

```python
np.linalg.eig(A)
```

---

# ✅ SECTION E — WORLD OF DATA & DESCRIPTIVE INSIGHTS (Q37–Q44)

Descriptive statistics summarize data using mean, median, variance, spread, etc. ([GeeksforGeeks][1])

---

### Q37. What is descriptive statistics?

**Answer:** Summarizing and organizing raw data.

---

### Q38. Measures of central tendency?

**Answer:** Mean, Median, Mode.

---

### Q39. Mean formula?

```python
sum(x)/n
```

---

### Q40. Median meaning?

**Answer:** Middle value after sorting.

---

### Q41. Mode meaning?

**Answer:** Most frequent value.

---

### Q42. Which is affected by outliers most?

**Answer:** Mean.

---

### Q43. Which is robust to outliers?

**Answer:** Median.

---

### Q44. What is percentile?

**Answer:** Position below which certain % data lies.

---

# ✅ SECTION F — DISPERSION INSIGHTS (Q45–Q50)

---

### Q45. What is dispersion?

**Answer:** Spread of data.

---

### Q46. Range formula?

```python
max-min
```

---

### Q47. Variance meaning?

**Answer:** Average squared deviation from mean.

---

### Q48. Standard deviation?

**Answer:** Square root of variance.

---

### Q49. High SD means?

**Answer:** Data highly spread.

---

### Q50. Low SD means?

**Answer:** Data close to mean.

---

# ✅ SECTION G — POWER OF ASSOCIATION (Q51–Q54)

---

### Q51. What is covariance?

**Answer:** Joint variability of two variables.

---

### Q52. What is correlation?

**Answer:** Strength of relationship between two variables.

---

### Q53. Correlation range?

**Answer:** -1 to +1

---

### Q54. Positive correlation means?

**Answer:** One increases, other increases.

---

# ✅ SECTION H — SKEWNESS (Q55–Q57)

Skewness measures asymmetry of distribution. ([GeeksforGeeks][1])

---

### Q55. Positive skew?

**Answer:** Tail on right side.

---

### Q56. Negative skew?

**Answer:** Tail on left side.

---

### Q57. Normal distribution skewness?

**Answer:** Approximately 0.

---

# ✅ SECTION I — PROBABILITY + DISTRIBUTIONS (Q58–Q65)

---

### Q58. Probability formula?

```python
Favorable / Total outcomes
```

---

### Q59. Probability range?

**Answer:** 0 to 1

---

### Q60. Uniform distribution?

**Answer:** All outcomes equally likely.

---

### Q61. Example of uniform?

**Answer:** Fair dice.

---

### Q62. Binomial distribution?

**Answer:** Repeated yes/no trials.

---

### Q63. Formula of binomial probability?

[
P(X)= {n \choose x} p^x (1-p)^{n-x}
]

---

### Q64. Example of binomial?

**Answer:** Tossing coin 10 times.

---

### Q65. Bernoulli vs Binomial?

**Answer:**
Bernoulli = one trial
Binomial = many Bernoulli trials

---

# ✅ SECTION J — Z SCORE & T SCORE (Q66–Q71)

---

### Q66. What is Z-score?

Distance of value from mean in SD units.

genui{"math_block_widget_always_prefetch_v2":{"content":"z=\frac{x-\mu}{\sigma}"}}

---

### Q67. Why use Z-score?

**Answer:** Standardization.

---

### Q68. What is T-score?

Used when sample size small and population SD unknown.

t=\frac{\bar{x}-\mu}{s/\sqrt{n}}

---

### Q69. When use Z test?

**Answer:** n > 30 usually.

---

### Q70. When use T test?

**Answer:** n < 30.

---

### Q71. T distribution shape?

**Answer:** Similar to normal but wider tails.

---

# ✅ SECTION K — INFERENTIAL STATISTICS (Q72–Q75)

Inferential statistics uses sample data to infer population conclusions. ([GeeksforGeeks][1])

---

### Q72. What is inferential statistics?

**Answer:** Making predictions about population from sample.

---

### Q73. Population vs Sample?

**Answer:**
Population = complete data
Sample = subset

---

### Q74. Parameter vs Statistic?

**Answer:**
Parameter from population
Statistic from sample

---

### Q75. Confidence interval meaning?

**Answer:** Estimated range for population parameter.

---

# ✅ SECTION L — HYPOTHESIS TESTING + P VALUE + CLT + ERRORS (Q76–Q80)

---

### Q76. What is Null Hypothesis?

**Answer:** Assumes no effect/no difference.

---

### Q77. What is Alternative Hypothesis?

**Answer:** Assumes effect exists.

---

### Q78. What is P-value?

Probability of getting result if null hypothesis is true.

---

### Q79. Rule of p-value?

**Answer:**
If p < 0.05 → reject H0
If p > 0.05 → fail to reject H0

---

### Q80. What is Central Limit Theorem?

Sample means tend to become normally distributed as sample size increases.

---
