# CART Algorithm Implementation

## Project Overview
This project focuses on the manual implementation of the **Classification and Regression Trees (CART)** algorithm in Python. The goal was to build a Decision Tree classifier from scratch using **Gini Impurity** as the splitting criterion and compare its performance against the industry-standard implementation from `scikit-learn`.

---

## 1. About CART
**CART (Classification and Regression Trees)** is a predictive modeling algorithm used in machine learning. It builds a decision tree by recursively splitting the dataset into subsets based on the value of input features.

* **How it works:** The algorithm searches for the best feature and threshold to split data to maximize "purity" in the resulting nodes.
* **The Metric:** In this classification implementation, we use **Gini Impurity** to measure the likelihood of incorrect classification of a new instance.
* **The Goal:** To create a tree where the leaves (end nodes) contain data points that are as homogeneous as possible regarding the target variable.

---

## 2. Project Requirements Met
This repository adheres to the following academic and technical guidelines:
* **PyScaffold Structure:** Organized directory layout for Python projects.
* **Custom Algorithm Implementation:** The CART logic is written from scratch (no `sklearn` used for the core logic).
* **Library Comparison:** The custom model is benchmarked against `sklearn.tree.DecisionTreeClassifier`.
* **PEP 8 Compliance:** Clean, readable, and standard Python code style.
* **Documentation:** Comprehensive documentation of the logic and usage (this file).

---

## 3. ⚙️ Core Components & Architecture
The custom solution is encapsulated in the `SimpleCART` class. Here are the key functions driving the logic:

* `fit(X, y)`: The entry point for training. It accepts features and labels and initiates the recursive tree-building process.
* `_build_tree(X, y, depth)`: A recursive function that constructs the nodes. It stops when the max depth is reached or the node is pure.
* `_best_split(X, y)`: The "brain" of the algorithm. It iterates through features to find the optimal threshold that minimizes Gini Impurity. *Includes an optimization to sample thresholds for continuous variables to improve speed.*
* `gini_impurity(y)`: A mathematical helper function that calculates the impurity of a specific node.
* `predict(X)`: Traverses the built tree to output class labels for new, unseen data.

---

## 4. Data Processing Pipeline
To test the algorithm, we utilize real-world data regarding global airports (`airports.csv`).

* **Dataset Size:** To ensure efficient training during development, the dataset is downsampled to **3,000 samples**.
* **Target Variable:** We classify airports into three types: `small_airport`, `medium_airport`, and `large_airport`.
* **Feature Engineering:**
    * **Geography:** Latitude, Longitude, and Elevation.
    * **Categorical Handling:** `Continent` and `ISO Country` are processed using One-Hot Encoding (Top 10 countries kept, others grouped).
* **Split Strategy:** The data is split into **Training (80%)** and **Testing (20%)** sets. Crucially, we use a **stratified split** to ensure the proportion of airport types remains consistent across both sets.

---

## 5. Results & Comparative Analysis
The project concludes with a direct comparison between the `SimpleCART` (Custom) and `Scikit-Learn` models.

### Performance Metrics
We evaluate the models based on:
1.  **Accuracy Score:** The percentage of correctly predicted airport types.
2.  **Confusion Matrix:** A detailed breakdown showing where the models got confused (e.g., misclassifying a medium airport as a small one).

### Observations
* **Accuracy:** The custom implementation achieves results highly comparable to the Scikit-Learn library.
* **Optimization:** The `SimpleCART` uses a stochastic threshold approach (sampling 20 thresholds for dense features). While Scikit-Learn is more exhaustive, the custom model provides a great balance between **computational speed** and **predictive accuracy**.
* **Constraints:** Both models are limited to a `max_depth=7` to prevent overfitting and ensure a fair comparison.
