
## **Section 1: Advanced Python (Functions, OOPs, Exception Handling)**  
### **Functions (7 Questions)**  
1. **Memoization for Fibonacci**  
   Write a recursive Fibonacci function with memoization to optimize performance.  
   **Input:** `10`  
   **Output:** `55`  

2. **Decorator to Measure Execution Time**  
   Create a decorator `@timeit` that prints the time taken by a function to execute.  

3. **Currying Function**  
   Write a curried function `add` such that `add(2)(3)(5)` returns `10`.  

4. **Closure for Counter**  
   Create a function `counter()` that returns a closure to increment and return a count.  
   **Usage:**  
   ```python
   c = counter()  
   print(c())  # 1  
   print(c())  # 2  
   ```  

5. **Function to Generate Prime Numbers**  
   Write a generator function `primes()` that yields prime numbers indefinitely.  

6. **Partial Function Application**  
   Use `functools.partial` to create a function `pow_2` that always raises numbers to the power of 2.  

7. **Dynamic Function Creation**  
   Write a function `create_multiplier(n)` that returns a lambda function to multiply by `n`.  

---

### **OOPs (7 Questions)**  
8. **Singleton Class**  
   Implement a Singleton class using `__new__`.  

9. **Method Resolution Order (MRO)**  
   Create a diamond inheritance structure (`A -> B, A -> C -> D`) and print the MRO of class `D`.  

10. **Abstract Base Class (ABC)**  
    Define an abstract class `Shape` with `area()` and `perimeter()` methods. Implement `Circle` and `Rectangle`.  

11. **Operator Overloading**  
    Overload `+` and `*` for a `Vector` class to support vector addition and scalar multiplication.  

12. **Property Decorator**  
    Create a class `Temperature` with a property `celsius` that automatically converts to/from `fahrenheit`.  

13. **Class Method & Static Method**  
    Write a class `StringUtils` with:  
    - A class method `from_csv(cls, csv_str)` to parse CSV.  
    - A static method `is_palindrome(s)` to check palindromes.  

14. **Descriptor Protocol**  
    Implement a `PositiveNumber` descriptor to enforce positive values in class attributes.  

---

### **Exception Handling (6 Questions)**  
15. **Custom Exception**  
    Define a custom exception `InvalidAgeError` and raise it if age < 0.  

16. **Context Manager for File Handling**  
    Create a context manager `FileHandler` that automatically closes files and handles `IOError`.  

17. **Retry Decorator**  
    Write a decorator `@retry(max_attempts=3)` that retries a function if it raises an exception.  

18. **Exception Chaining**  
    Demonstrate exception chaining by raising a `ValueError` from a `KeyError`.  

19. **Suppress Specific Exceptions**  
    Use `contextlib.suppress` to ignore `ZeroDivisionError` in a block.  

20. **Database Transaction Rollback**  
    Simulate a database transaction with `commit()` and `rollback()` using exceptions.  

---

## **Section 2: NumPy & Pandas (20 Questions - Pick any dataset from Kaggle)**  
### **NumPy (10 Questions)**  
21. **Boolean Indexing**  
    Extract all elements > 5 from a NumPy array.  
    **Input:** `np.array([1, 6, 2, 8, 3])`  
    **Output:** `[6, 8]`  

22. **Broadcasting**  
    Subtract the mean of each row from a 2D array.  

23. **Matrix Multiplication**  
    Multiply two matrices without using `np.dot()`.  

24. **Structured Arrays**  
    Create a structured array to store `(name, age, salary)` and sort by salary.  

25. **Vectorized Operations**  
    Replace all negative values in an array with 0 using `np.where`.  

26. **Linear Algebra**  
    Solve the system of equations:  
    ```  
    3x + y = 9  
    x + 2y = 8  
    ```  

27. **Random Sampling**  
    Generate 10 random dates in 2023 using `np.datetime64`.  

28. **Masked Arrays**  
    Compute the mean of an array while ignoring negative values.  

29. **Advanced Indexing**  
    Use fancy indexing to extract rows 1, 3, and 5 from a 2D array.  

30. **Universal Functions (ufuncs)**  
    Create a custom ufunc to compute the sigmoid function.  

---

### **Pandas (10 Questions)**  
31. **Data Cleaning**  
    Fill missing values in a DataFrame with the median of each column.  

32. **GroupBy Aggregation**  
    Compute the average salary by department in a DataFrame.  

33. **Pivot Tables**  
    Create a pivot table showing total sales by `region` and `product`.  

34. **Time Series Analysis**  
    Resample a time series DataFrame to monthly frequency and compute the mean.  

35. **Merge/Join**  
    Merge two DataFrames on a key column with an `outer` join.  

36. **Apply Function**  
    Use `apply()` to categorize ages into bins (e.g., "Child", "Adult").  

37. **MultiIndexing**  
    Create a MultiIndex DataFrame and slice data for a specific level.  

38. **String Operations**  
    Extract all email domains from a column using `str.extract()`.  

39. **Rolling Window**  
    Compute the 7-day rolling average of stock prices.  

40. **Optimization**  
    Convert a DataFrame column from strings to categories to save memory.  

---

### **Section 3: Machine Learning (Supervised Learning) – Kaggle Datasets - Any 5**  
**Each question includes:**  
- **Task**  
- **Kaggle Dataset Link**  
- **Key Libraries** (`scikit-learn`, `XGBoost`, `TensorFlow`, etc.)  

---

### **1. Linear Regression**  
**Task:** Predict house prices using feature engineering.  
**Dataset:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
**Key Steps:**  
- Handle missing values (`SimpleImputer`).  
- Encode categorical variables (`OneHotEncoder`).  
- Use `Ridge`/`Lasso` regression to avoid overfitting.  

---

### **2. Logistic Regression**  
**Task:** Predict Titanic survival with feature selection.  
**Dataset:** [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)  
**Key Steps:**  
- Impute missing `Age` values.  
- Use `SelectKBest` for feature selection.  
- Optimize `LogisticRegression` with `GridSearchCV`.  

---

### **3. Decision Trees**  
**Task:** Classify iris species and visualize the tree.  
**Dataset:** [Iris Species](https://www.kaggle.com/uciml/iris)  
**Key Steps:**  
- Train `DecisionTreeClassifier`.  
- Plot the tree using `plot_tree`.  
- Prune the tree using `ccp_alpha`.  

---

### **4. Random Forest**  
**Task:** Predict breast cancer malignancy and analyze feature importance.  
**Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)  
**Key Steps:**  
- Use `RandomForestClassifier`.  
- Plot feature importances with `matplotlib`.  

---

### **5. SVM (Support Vector Machines)**  
**Task:** Classify handwritten digits (MNIST).  
**Dataset:** [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)  
**Key Steps:**  
- Scale features using `StandardScaler`.  
- Use `SVC(kernel='rbf')`.  
- Reduce dimensionality with `PCA` for speed.  

---

### **6. k-NN (k-Nearest Neighbors)**  
**Task:** Predict wine class and optimize `k`.  
**Dataset:** [Wine Dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)  
**Key Steps:**  
- Normalize data with `MinMaxScaler`.  
- Find optimal `k` using cross-validation.  

---

### **7. Gradient Boosting (XGBoost)**  
**Task:** Predict diabetes progression.  
**Dataset:** [Pima Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
**Key Steps:**  
- Handle class imbalance (`scale_pos_weight`).  
- Tune hyperparameters with `RandomizedSearchCV`.  

---

### **8. Neural Networks (MLP)**  
**Task:** Binary classification for credit card fraud.  
**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Key Steps:**  
- Use `MLPClassifier` or `TensorFlow/Keras`.  
- Address imbalance with `SMOTE` or class weights.  

---

### **9. Time Series Forecasting (LSTM)**  
**Task:** Predict stock prices using LSTM.  
**Dataset:** [Tesla Stock Price](https://www.kaggle.com/rpaguirre/tesla-stock-price)  
**Key Steps:**  
- Reshape data for LSTM input (`timesteps=60`).  
- Use `TensorFlow`’s `LSTM` layer.  

---

### **10. NLP (Sentiment Analysis)**  
**Task:** Classify movie reviews as positive/negative.  
**Dataset:** [IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
**Key Steps:**  
- Text preprocessing (`TF-IDF`/`Word2Vec`).  
- Train `LogisticRegression` or `BERT`.  

---

### **11. Feature Engineering**  
**Task:** Improve house price predictions using feature creation.  
**Dataset:** [House Prices (Advanced)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
**Key Steps:**  
- Create interaction features (e.g., `TotalSF = 1stFlrSF + 2ndFlrSF`).  
- Use `PolynomialFeatures`.  

---

### **12. Hyperparameter Tuning**  
**Task:** Optimize a Random Forest for the Titanic dataset.  
**Dataset:** [Titanic](https://www.kaggle.com/c/titanic/data)  
**Key Steps:**  
- Use `Optuna` or `BayesianOptimization`.  
- Tune `max_depth`, `n_estimators`, etc.  

---

### **13. Model Interpretability (SHAP)**  
**Task:** Explain predictions for the Breast Cancer dataset.  
**Dataset:** [Breast Cancer Wisconsin](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)  
**Key Steps:**  
- Use `SHAP` to explain feature contributions.  
- Visualize with `summary_plot`.  

---

### **14. Ensemble Learning**  
**Task:** Combine XGBoost and Logistic Regression using stacking.  
**Dataset:** [Titanic](https://www.kaggle.com/c/titanic/data)  
**Key Steps:**  
- Use `StackingClassifier` from `scikit-learn`.  

---

### **15. Cross-Validation Strategies**  
**Task:** Evaluate a model using stratified K-fold for imbalanced data.  
**Dataset:** [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Key Steps:**  
- Use `StratifiedKFold` with `LogisticRegression`.  

---

### **16. Dimensionality Reduction (PCA)**  
**Task:** Reduce features for the Wine dataset and visualize clusters.  
**Dataset:** [Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)  
**Key Steps:**  
- Apply `PCA` and plot clusters in 2D.  

---

### **17. AutoML (PyCaret)**  
**Task:** Automate model selection for the Pima Diabetes dataset.  
**Dataset:** [Pima Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
**Key Steps:**  
- Use `PyCaret` to compare models in 3 lines of code.  

---

### **18. Custom Loss Function**  
**Task:** Implement a custom loss function for house price prediction.  
**Dataset:** [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
**Key Steps:**  
- Use `XGBoost` with `custom objective`.  

---

### **19. Deploy a Model (Flask)**  
**Task:** Deploy a Titanic survival predictor as a web API.  
**Dataset:** [Titanic](https://www.kaggle.com/c/titanic/data)  
**Key Steps:**  
- Save model with `pickle`.  
- Create a Flask endpoint.  

---

### **20. Bias-Variance Tradeoff**  
**Task:** Diagnose overfitting for the Iris dataset.  
**Dataset:** [Iris](https://www.kaggle.com/uciml/iris)  
**Key Steps:**  
- Plot learning curves (`train_score` vs `val_score`).  
