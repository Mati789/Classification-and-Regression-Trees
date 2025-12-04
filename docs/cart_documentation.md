# Project Technical Documentation

## Overview: Goal and Methodology

This project implements and analyzes the **Classification and Regression Tree (CART)** algorithm. The goal is to predict the number of flight stopovers (`Total_Stops`).

The core methodology:
1.  **Demonstrate the mathematical foundation** using **Entropy** and **Information Gain**.
2.  **Compare two models:** a simple, restricted model vs. a complex, high-accuracy model.

The analysis uses flight data from **'dataset.csv'** with **Pandas** and **Scikit-learn**.

---
## 1. Key Parameters and Data Setup

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Data Utilized** | First 5000 Rows | Fixed, consistent dataset size. |
| **Data Split** | 90% Training / 10% Testing | High training ratio (4500 rows) for effective pattern learning. |
| **Features (X)** | `Airline`, `Source`, `Destination` | Three categorical input columns. |
| **Target (Y)** | `Total_Stops` | The predicted categorical variable. |
| **Model 1: max_depth** | 3 (Restricted) | **Guarantees** the model is simple and ensures lower accuracy than the full tree. |
| **Splitting Criterion** | Entropy / Information Gain | Metric used to select the best feature for node splitting. |
| **Pre-processing** | Label Encoding | Converts all categorical data into integer values. |

---
## 2. Analysis of Model Results

### 2.1. Interpretation of Entropy and Information Gain (IG)

* **Training Set Entropy (e.g., 1.4462):** Measures the initial **impurity** (randomness) of the target classes. A value closer to 2.0 (max for 4 classes) indicates high uncertainty.
* **Information Gain (IG):** Measures the **reduction in uncertainty** achieved by a split. The feature with the highest IG is selected as the best splitter. 
    * **Conclusion:** The feature with the highest IG (e.g., **`Destination`**) is confirmed as the most critical factor for prediction.

### 2.2. Model Performance Comparison

| Metric | Simple CART (Entropy, Max Depth 3) | Scikit-learn Comparison (Gini, Full Depth) |
| :--- | :--- | :--- |
| **Typical Accuracy** | Significantly lower (e.g., 0.75 - 0.80) | Significantly higher (e.g., 0.85 - 0.90) |
| **Goal** | Show the effect of **model constraint** and simplicity. | Show the maximum performance achievable with current features. |

### 2.3. Confusion Matrix (CM) Analysis

The CM is **4x4** because the target variable (`Total_Stops`) has four unique classes.

* **Simple Model CM:** Shows high error rates off the main diagonal. **Shallow depth** prevents learning complex rules for minority classes (e.g., `3 stops`).
* **Comparison Model CM:** Shows much higher correct prediction rates on the main diagonal. **Greater complexity** allows for superior classification across all classes.

---
## 3. Key Python Functions Breakdown

### 3.1. `load_and_preprocess_data()`

* **Purpose:** Loads data, cleans missing values, and prepares data for modeling.
* **Key Action:** Executes **Label Encoding** on all features and the target.

### 3.2. `calculate_entropy(y_data)`

* **Purpose:** Computes the mathematical uncertainty of any given data subset.
* **Mechanism:** Applies the formula $H(X) = -\sum P(x_i) \log_2(P(x_i))$.

### 3.3. `calculate_information_gain(X_data, y_data, feature_col)`

* **Purpose:** Determines which feature provides the best split.
* **Mechanism:** Calculates the reduction in entropy. The highest IG wins.
