## **Detailed Lecture Notes: Decision Trees for Classification - From Rules to Models**

**Guide:** Vineet Tiwari
**Course:** Advanced Data Analysis and Statistical Inference
**Lecture Topic:** Building Interpretable Tree-Based Classifiers and Controlling Overfitting

---

### **1. Introduction: Models That Mirror Human Decision-Making**

After exploring linear and logistic models, we now turn to a fundamentally different approach: **Decision Trees**. Unlike previous models that create a global equation, a decision tree partitions the feature space into a series of rectangular regions using simple, hierarchical "if-else" rules. This structure is highly intuitive and mirrors human decision-making processes (e.g., "If the customer's age is less than 35 *and* their income is high, then they are likely to purchase").

**Key Advantages:**
*   **Interpretability:** The model can be visualized and understood easily, even by non-experts.
*   **Nonlinearity & Interactions:** They automatically capture complex nonlinear relationships and interactions between features without needing manual feature engineering (like polynomial terms in regression).
*   **Data Flexibility:** They handle both numerical and categorical data seamlessly and require little data preprocessing (e.g., no need for standardization).
*   **Robustness:** They are insensitive to outliers and monotonic transformations of the features.

**The Fundamental Challenge:** Their flexibility makes them highly prone to **overfitting**. A tree that is grown too deep will create overly specific rules that perfectly fit the training data but fail to generalize to new data. The art of building a decision tree lies in controlling this complexity.

---

### **2. Core Concepts: How a Tree Grows**

A decision tree is built by recursively splitting the data into purer and purer subsets.

*   **The Process (Recursive Partitioning):**
    1.  **Start:** Begin at the root node with the entire dataset.
    2.  **Find the Best Split:** For every feature, and for every possible threshold (for continuous data) or category (for categorical data), calculate how well that split separates the classes.
    3.  **Split:** Choose the single feature and threshold that results in the **greatest increase in purity** in the resulting child nodes.
    4.  **Repeat:** Recursively apply steps 2-3 to each child node until a **stopping criterion** is met.

*   **Measuring Purity (Node Impurity):** We need a metric to quantify how "mixed" the classes are in a node. Two common measures are:
    *   **Gini Impurity:** Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution in the node. A node with only one class (pure) has a Gini Impurity of 0.
        `$G = 1 - \sum_{k=1}^{K} (p_k)^2$`
    *   **Entropy / Information Gain:** Entropy measures the disorder in a node. **Information Gain** is the reduction in entropy after a split. We choose the split that maximizes information gain.
        `$E = - \sum_{k=1}^{K} p_k \log_2(p_k)$`
    *   In practice, for classification, Gini and Entropy often yield very similar results. Gini is slightly faster to compute and is the default in `scikit-learn`.

*   **Stopping and Pruning:** Letting a tree grow until all nodes are pure is a recipe for overfitting. We control this by:
    *   **Pre-Pruning (Stopping Criteria):** Setting hyperparameters that stop growth early (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`).
    *   **Post-Pruning (Cost Complexity Pruning):** Growing a large tree first and then "pruning" it back by removing branches that provide the least predictive power, using a complexity parameter `ccp_alpha`.

---

### **3. Worked Example in Python: The Overfitting Problem and its Solution**

The following code demonstrates the entire process: creating a complex dataset, building an overfit tree, and then applying constraints to build a more generalizable model.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate a complex synthetic dataset with 4 informative features out of 10
X, y = make_classification(n_samples=1500, n_features=10, n_informative=4,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           class_sep=1.2, weights=[0.6, 0.4], random_state=7)
# Create a DataFrame for better interpretation (optional)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
class_names = ['Class_0', 'Class_1']

# Split the data, ensuring stratification due to class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- Model 1: A Baseline, Unconstrained Tree (WILL OVERFIT) ---
tree_base = DecisionTreeClassifier(criterion='gini', random_state=42)
tree_base.fit(X_train, y_train)

# Diagnose Overfitting: Look at the massive size of the tree
print("=== Baseline (Overfit) Tree ===")
print(f"Tree Depth: {tree_base.get_depth()}")
print(f"Number of Leaves: {tree_base.get_n_leaves()}\n")

# Evaluate on Training vs. Test data -> The gap indicates overfitting
print(f"Training Score: {tree_base.score(X_train, y_train):.4f}")
print(f"Test Score: {tree_base.score(X_test, y_test):.4f}")
# The test score will be significantly lower than the training score.

print('\nClassification Report (Base Tree - Test Set):')
print(classification_report(y_test, tree_base.predict(X_test), target_names=class_names, digits=3))

# --- Model 2: A Constrained, Regularized Tree (GENERALIZES BETTER) ---
tree_tuned = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,           # Prevents the tree from growing too deep
    min_samples_split=20,  # Node must have at least 20 samples to be split
    min_samples_leaf=10,   # A leaf must have at least 10 samples
    random_state=42
)
tree_tuned.fit(X_train, y_train)

print("\n=== Constrained (Tuned) Tree ===")
print(f"Tree Depth: {tree_tuned.get_depth()}")
print(f"Number of Leaves: {tree_tuned.get_n_leaves()}\n")

print(f"Training Score: {tree_tuned.score(X_train, y_train):.4f}")
print(f"Test Score: {tree_tuned.score(X_test, y_test):.4f}")
# The gap between training and test performance should be smaller.

print('\nClassification Report (Tuned Tree - Test Set):')
print(classification_report(y_test, tree_tuned.predict(X_test), target_names=class_names, digits=3))

# --- Performance Comparison: ROC Curve ---
# Get predicted probabilities for the positive class
y_proba_base = tree_base.predict_proba(X_test)[:, 1]
y_proba_tuned = tree_tuned.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_base = roc_auc_score(y_test, y_proba_base)
auc_tuned = roc_auc_score(y_test, y_proba_tuned)
print(f'\nROC AUC (Base Tree): {auc_base:.3f}')
print(f'ROC AUC (Tuned Tree): {auc_tuned:.3f}')

# Plot ROC Curves
fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)
fpr_t, tpr_t, _ = roc_curve(y_test, y_proba_tuned)

plt.figure(figsize=(8, 6))
plt.plot(fpr_b, tpr_b, label=f'Base Tree (AUC = {auc_base:.3f})', lw=2)
plt.plot(fpr_t, tpr_t, label=f'Tuned Tree (AUC = {auc_tuned:.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison: Overfit vs. Regularized Tree')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- Model Interpretability: Visualizing the Tree ---
# Visualize the top 3 levels of the constrained tree
plt.figure(figsize=(16, 10))
plot_tree(tree_tuned,
          filled=True,            # Color nodes by majority class
          feature_names=feature_names,
          class_names=class_names,
          max_depth=3,            # Only show the first 3 splits for clarity
          proportion=True,        # Show proportions instead of raw counts
          rounded=True,
          fontsize=10)
plt.title('Decision Tree (Top 3 Levels) - Filled colors indicate majority class')
plt.tight_layout()
plt.show()

# Extract a text-based rule for a specific path (incredibly useful for deployment)
tree_rules = export_text(tree_tuned, feature_names=feature_names, max_depth=3)
print("\nText-based representation of the tree (max_depth=3):")
print(tree_rules)
```

**Code Walkthrough and Teaching Notes:**
*   **The Overfit Tree:** The baseline tree is grown without constraints. It achieves near-perfect training accuracy but has a significant drop in test accuracy. The large number of leaves indicates it has memorized the noise in the training data.
*   **The Constrained Tree:** By limiting the depth (`max_depth=5`) and requiring a minimum number of samples to split a node (`min_samples_split=20`) or form a leaf (`min_samples_leaf=10`), we build a simpler, more robust model. The test performance of this model is often *better* than the overfit tree, demonstrating the **bias-variance tradeoff**.
*   **ROC-AUC Comparison:** The Area Under the ROC Curve provides a single number to compare model performance across all classification thresholds. The tuned tree often has a higher AUC because it generalizes better.
*   **Interpretability:** The `plot_tree` function and `export_text` are invaluable. They allow us to see the exact decision rules, making the model's predictions transparent and auditable. This is a key advantage over "black box" models.

---

### **4. Beyond the Basics: Interpretability and Feature Importance**

*   **Feature Importance:** Decision trees provide a natural way to rank features by their importance. The importance of a feature is computed as the (normalized) total reduction in the impurity criterion (Gini/Entropy) brought by that feature across all splits in the tree.
    `feature_importances_ = tree_tuned.feature_importances_`
    *   **Caution:** This measure is biased towards features with more levels (high cardinality) and features that are used near the top of the tree.

*   **Partial Dependence Plots (PDPs):** Even for a single tree, we can use PDPs to visualize the marginal effect of a feature on the predicted outcome. This helps understand the model's behavior beyond the simple rules.

---

### **5. Strengths and Limitations: A Balanced View**

| Strengths | Limitations |
| :--- | :--- |
| **Highly Interpretable** | **High Variance:** Small changes in data can lead to a completely different tree (unstable). |
| **No Data Preprocessing** | **Greedy Algorithm:** Makes locally optimal splits at each node, not guaranteed to be globally optimal. |
| **Handles Nonlinearity** | **Poor Extrapolation:** Struggles to predict outside the range of the training data. |
| **Handles Mixed Data Types** | **Lower Predictive Accuracy:** Typically outperformed by ensemble methods (Random Forests, Gradient Boosting). |

---

### **6. Key Takeaways**

1.  **Interpretability is Key:** Decision trees are powerful because their logic is transparent. They are an excellent choice when explaining the model is as important as its accuracy.
2.  **The Enemy is Overfitting:** A tree's flexibility is its greatest weakness. You **must** control complexity through pre-pruning (`max_depth`, `min_samples_*`) or post-pruning (`ccp_alpha`) to build a useful model.
3.  **Evaluate Holistically:** Don't just look at accuracy. Use ROC-AUC, examine the confusion matrix, and most importantly, **always compare performance on a held-out test set** to diagnose overfitting.
4.  **A Building Block:** While powerful on their own, their true potential is realized when combined into **ensembles** like Random Forests, which mitigate their high variance.

---

### **7. Next Lecture Preview**

To solve the stability and accuracy limitations of a single tree, we combine many of them.

**Next Lecture: Random Forests and Ensemble Methods**

*   **The Core Idea: Bagging (Bootstrap Aggregating):** Learn how building many trees on different random subsets of the data and averaging their predictions reduces variance and creates a stronger, more stable model.
*   **The "Random" Part:** Understand how random feature selection during splitting de-correlates the trees, making the ensemble much more powerful than the sum of its parts.
*   **Out-of-Bag (OOB) Error:** Use a clever trick to get a nearly free validation score without a separate test set.
*   **Enhanced Feature Importance:** See how Random Forests provide more reliable feature importance scores.

**Are there any questions on the fundamentals of Decision Trees before we move to ensembles?**