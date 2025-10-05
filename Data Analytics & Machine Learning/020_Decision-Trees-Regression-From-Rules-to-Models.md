## **Detailed Lecture Notes: Decision Trees for Regression - Predicting Continuous Outcomes**

**Guide:** Vineet Tiwari
**Course:** Advanced Data Analysis and Statistical Inference
**Lecture Topic:** From Classification to Regression: Adapting Tree-Based Models for Continuous Responses

---

### **1. Introduction: Predicting Numbers Instead of Categories**

In our previous lecture, we explored how decision trees can partition the feature space to classify data into distinct categories (e.g., "Spam"/"Not Spam"). The core idea was to create pure nodes where one class dominates.

Now, we turn our attention to **regression** problems, where the target variable we wish to predict is **continuous** (e.g., house prices, stock values, patient blood pressure). The fundamental structure of the tree remains the same—a series of hierarchical, if-else rules that split the data. However, the objective and the mechanics for making predictions change significantly.

**Analogy:**
*   **Classification Tree:** A doctor's flowchart for diagnosing a disease (Output: Disease A, B, or C).
*   **Regression Tree:** A real estate agent's method for pricing a house.
    *   *If* square footage > 2000, *and* if zip code is in a premium area, *and* if age of home < 10 years, *then* the price is likely to be **around $750,000**.

The tree doesn't output a class probability but a **constant value**—typically the average—of the training samples in the leaf node.

**Key Takeaway:** A Regression Tree partitions the feature space into distinct, non-overlapping **segments** and models the response as a simple constant (the mean) within each segment.

---

### **2. Core Concepts: The Regression Tree Algorithm**

The process of growing a regression tree is identical to a classification tree in its structure: **Recursive Binary Partitioning**. The critical difference lies in the criterion used to choose the best split.

#### **Splitting Criterion: Minimizing Variance**

For a regression problem, the concept of "node impurity" is replaced by **node variance** or **MSE (Mean Squared Error)**. A "good" or "pure" node in regression is one where the values of the target variable are very close to each other, i.e., the variance is low.

*   **Prediction in a Node:** For any given node \( R_m \), the predicted value for a new data point that falls into that node is simply the **mean** of the training observations in \( R_m \).
    `$ \hat{y}_{R_m} = \frac{1}{n_m} \sum_{i \in R_m} y_i $`

*   **Objective Function:** The algorithm seeks splits that minimize the **Sum of Squared Errors (SSE)** within the resulting child nodes. For any candidate split that partitions the data into two regions, \( R_1 \) and \( R_2 \), we calculate the total SSE as:
    `$ SSE = \sum_{i \in R_1} (y_i - \hat{y}_{R_1})^2 + \sum_{i \in R_2} (y_i - \hat{y}_{R_2})^2 $`
    The split that results in the **largest possible reduction in SSE** is chosen. This is equivalent to minimizing the weighted average variance of the two new nodes.

**Comparison with Classification:**

| Aspect | Classification Tree | Regression Tree |
| :--- | :--- | :--- |
| **Splitting Criterion** | Minimize **Impurity** (Gini, Entropy) | Minimize **Variance / MSE** |
| **Leaf Node Prediction** | **Majority Class** (or class probability) | **Mean Value** of the target |
| **Model Output** | Class Label / Probability | Continuous Numerical Value |

#### **Controlling Overfitting: The Same Enemy, Same Defenses**

Just like their classification counterparts, regression trees are highly prone to overfitting. An unconstrained tree will keep splitting until each leaf contains only a few observations, perfectly modeling the training data (including noise) but failing to generalize.

The same pruning techniques apply:
*   **Pre-Pruning:** Using hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.
*   **Post-Pruning:** Using Cost-Complexity Pruning (`ccp_alpha`) to trim back a large tree to an optimal size.

---

### **3. Worked Example in Python: Predicting a Continuous Outcome**

Let's simulate a dataset where the relationship between the features and the target is nonlinear—a scenario where linear regression might struggle, but decision trees can excel.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set a random seed for reproducibility
np.random.seed(42)

# Simulate a nonlinear dataset: House Price based on Size and Location Score
n_samples = 500
size = np.random.uniform(500, 4000, n_samples)  # Square footage
location_score = np.random.uniform(1, 10, n_samples) # Arbitrary location score

# Create a nonlinear relationship with interaction
# True model: Price = base + (size * location_multiplier) + (size^2 effect) + noise
price = (20000 +
         100 * size * (location_score / 2) +  # Interaction effect
         0.05 * (size ** 2) +                 # Nonlinear effect
         np.random.normal(0, 20000, n_samples)) # Noise

# Create a DataFrame
df = pd.DataFrame({'Price': price, 'Size': size, 'Location_Score': location_score})

# Split the data
X = df[['Size', 'Location_Score']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Data Shapes:", X_train.shape, X_test.shape)
print(f"Average Price in Training Set: ${y_train.mean():.2f}")

# --- Model 1: Overfitting Regression Tree ---
tree_overfit = DecisionTreeRegressor(random_state=42)
# Let it grow without constraints - it will overfit!
tree_overfit.fit(X_train, y_train)

print("\n" + "="*55)
print("=== Overfitting Regression Tree ===")
print("="*55)
print(f"Tree Depth: {tree_overfit.get_depth()}")
print(f"Number of Leaves: {tree_overfit.get_n_leaves()}")

# Evaluate performance
y_pred_train_of = tree_overfit.predict(X_train)
y_pred_test_of = tree_overfit.predict(X_test)

mse_train_of = mean_squared_error(y_train, y_pred_train_of)
mse_test_of = mean_squared_error(y_test, y_pred_test_of)

print(f"\nTraining MSE: {mse_train_of:,.2f}")
print(f"Test MSE: {mse_test_of:,.2f}")
print(f"MSE Ratio (Test/Train): {mse_test_of/mse_train_of:.2f}") # A large ratio indicates overfitting

r2_train_of = r2_score(y_train, y_pred_train_of)
r2_test_of = r2_score(y_test, y_pred_test_of)
print(f"\nTraining R²: {r2_train_of:.4f}") # Will be very close to 1
print(f"Test R²: {r2_test_of:.4f}")       # Will be significantly lower

# --- Model 2: Regularized Regression Tree ---
tree_tuned = DecisionTreeRegressor(
    max_depth=4,               # Limit the depth
    min_samples_split=20,      # Require at least 20 samples to split a node
    min_samples_leaf=10,       # Require at least 10 samples in a leaf
    random_state=42
)
tree_tuned.fit(X_train, y_train)

print("\n" + "="*55)
print("=== Regularized (Tuned) Regression Tree ===")
print("="*55)
print(f"Tree Depth: {tree_tuned.get_depth()}")
print(f"Number of Leaves: {tree_tuned.get_n_leaves()}")

# Evaluate performance
y_pred_train_tuned = tree_tuned.predict(X_train)
y_pred_test_tuned = tree_tuned.predict(X_test)

mse_train_tuned = mean_squared_error(y_train, y_pred_train_tuned)
mse_test_tuned = mean_squared_error(y_test, y_pred_test_tuned)

print(f"\nTraining MSE: {mse_train_tuned:,.2f}")
print(f"Test MSE: {mse_test_tuned:,.2f}")
print(f"MSE Ratio (Test/Train): {mse_test_tuned/mse_train_tuned:.2f}") # Closer to 1 is better

r2_train_tuned = r2_score(y_train, y_pred_train_tuned)
r2_test_tuned = r2_score(y_test, y_pred_test_tuned)
print(f"\nTraining R²: {r2_train_tuned:.4f}")
print(f"Test R²: {r2_test_tuned:.4f}") # This is often higher than the overfit tree's test R²

# --- Visualizing the Tree Structure ---
plt.figure(figsize=(20, 12))
plot_tree(tree_tuned,
          filled=True,
          feature_names=X.columns,
          proportion=True,
          rounded=True,
          precision=1,
          fontsize=10)
plt.title('Tuned Regression Tree Structure\n(Leaf nodes show the predicted price)', size=16)
plt.tight_layout()
plt.show()

# --- Text-Based Rules ---
print("\n" + "="*55)
print("Text-based Representation of the Tuned Tree:")
print("="*55)
tree_rules = export_text(tree_tuned, feature_names=list(X.columns), decimals=1)
print(tree_rules)

# --- Feature Importance ---
importances = tree_tuned.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
plt.title('Feature Importances from the Tuned Regression Tree')
plt.tight_layout()
plt.show()

# --- Visualizing the Piecewise Constant Predictions ---
# To understand how the tree works, let's see its prediction surface.

# Create a grid of feature values
size_range = np.linspace(X['Size'].min(), X['Size'].max(), 100)
loc_range = np.linspace(X['Location_Score'].min(), X['Location_Score'].max(), 100)
size_grid, loc_grid = np.meshgrid(size_range, loc_range)

# Predict for every point in the grid
grid_predictions = tree_tuned.predict(np.c_[size_grid.ravel(), loc_grid.ravel()])
grid_predictions = grid_predictions.reshape(size_grid.shape)

# Plot the prediction surface
plt.figure(figsize=(10, 6))
contour = plt.contourf(size_grid, loc_grid, grid_predictions, levels=20, alpha=0.8)
plt.colorbar(contour, label='Predicted Price')
plt.scatter(X_train['Size'], X_train['Location_Score'], c=y_train, s=20, edgecolor='white', label='Training Data')
plt.xlabel('Size (sq ft)')
plt.ylabel('Location Score')
plt.title('Regression Tree Prediction Surface\n(Piecewise Constant Regions)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Code Walkthrough and Teaching Notes:**

*   **The Overfit Tree (`tree_overfit`):**
    *   Grown without constraints, it achieves a near-perfect fit on the training data (R² ≈ 1.0, very low Training MSE). However, its performance on the test set is much worse.
    *   The `MSE Ratio (Test/Train)` will be a large number (e.g., 5, 10, or more), quantitatively demonstrating the overfitting. The model has memorized the noise.

*   **The Tuned Tree (`tree_tuned`):**
    *   By applying constraints (`max_depth=4`, etc.), we build a simpler model.
    *   **Key Result:** While the training performance is worse than the overfit tree, the **test performance is better**. The test R² for the tuned tree is often higher, and the MSE Ratio is much closer to 1. This is the essence of successful model regularization.

*   **Interpreting the Tree:**
    *   The `plot_tree` output shows that each leaf node contains a `value`. This `value` is the **mean Price** of all training samples that ended up in that leaf. This is the prediction for any new data point that follows that path.
    *   The `export_text` function gives the exact rules, such as `Price = 359384.5` for data points where `Size <= 2192.5` and `Location_Score <= 6.8`.

*   **The Piecewise Constant Surface:**
    *   The contour plot is the most illuminating visualization. It shows that the regression tree's prediction surface is not a smooth plane or curve. Instead, it is a series of **flat, rectangular plateaus**. Each plateau corresponds to a leaf node in the tree. This highlights a key characteristic (and limitation) of simple decision trees: they can only produce constant predictions within a region.

---

### **4. Strengths, Limitations, and Comparison to Linear Models**

| Aspect | Linear Regression | Regression Tree |
| :--- | :--- | :--- |
| **Model Form** | Global Linear Equation | Piecewise Constant Approximation |
| **Interpretability** | Highly interpretable coefficients. | Highly interpretable rules (if the tree is small). |
| **Handling Nonlinearity** | Poor, unless manual feature engineering (polynomials) is used. | Excellent, captures complex patterns automatically. |
| **Handling Interactions** | Manual inclusion of interaction terms (`X1 * X2`) is required. | Automatically detects and models interactions. |
| **Prediction Surface** | Smooth and continuous. | Jagged, piecewise constant, and discontinuous. |
| **Extrapolation** | Can extrapolate trends into the future. | Very poor; cannot predict outside the range of training data. |

**When to Choose a Regression Tree?**
*   When the relationship between features and target is highly nonlinear or involves complex interactions.
*   When interpretability is important, and you have a small-to-moderate number of features.
*   When you are less concerned with extrapolation and more with interpolation within the feature space.

---

### **5. Key Takeaways**

1.  **From Classes to Means:** The core difference from classification trees is the leaf node prediction. Regression trees predict the **mean** of the target variable in each leaf.
2.  **Splitting for Low Variance:** The splitting criterion changes from minimizing impurity (Gini/Entropy) to minimizing variance or MSE within the child nodes.
3.  **The Overfitting Problem Persists:** Unconstrained regression trees will overfit just as severely as classification trees. The same pruning strategies (`max_depth`, `min_samples_leaf`, etc.) are essential for building a generalizable model.
4.  **Piecewise Constant Model:** The overall model is a **piecewise constant function**. It approximates complex relationships by dividing the feature space into rectangles and assigning a simple, constant value to each one.
5.  **A Foundation for Ensembles:** Single regression trees are often not the most accurate predictors. However, they serve as the fundamental building block for powerful ensemble methods like **Random Forest Regressors** and **Gradient Boosted Regression Trees**, which combine many trees to create a strong, smooth, and highly accurate predictive model.

---

### **6. Next Lecture Preview**

The high variance of a single tree—evident in the jagged prediction surface—is a major limitation. The solution is to build an ensemble.

**Next Lecture: Random Forests and Gradient Boosting for Regression**

*   **A Forest of Trees:** We will extend the **Random Forest** algorithm to regression, learning how averaging the predictions of many de-correlated trees creates a smoother, more stable, and more accurate model.
*   **Boosting: Sequential Improvement:** We will introduce **Gradient Boosting Machines (GBM)**, a powerful technique where trees are built sequentially, with each new tree learning to correct the errors of the previous ones.
*   **Smoother Predictions:** See how these ensembles produce a much smoother prediction surface compared to the jagged output of a single tree.
*   **Hyperparameter Tuning:** Learn the key hyperparameters for these ensemble methods (e.g., `n_estimators`, `learning_rate`) and how to tune them for optimal performance.

**Are there any questions on how Decision Trees adapt to regression problems before we combine them into forests?**