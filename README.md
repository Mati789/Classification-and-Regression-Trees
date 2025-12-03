# CART Decision Tree Project Documentation

## 1. Project Overview and Objectives

This project implements the **Classification and Regression Tree (CART)** algorithm from scratch in Python, fulfilling the requirement for an **"Own Implementation"** of a core machine learning model.

### A. Key Objectives and Data Source
* **Goal:** To build a single decision tree model capable of classifying airport entries into one of three types: `small_airport`, `medium_airport`, or `large_airport`.
* **Data Source:** The project utilizes the **`airports.csv`** file for all training and testing data.
* **Comparison:** The custom model's performance (`SimpleCART`) is validated by direct comparison against the highly optimized **`DecisionTreeClassifier`** from the Scikit-learn library.

## 2. Technical Requirements and Compliance

| Requirement | Status | Explanation |
| :--- | :--- | :--- |
| **Own Implementation** | Compliant | The **`SimpleCART`** class provides a custom implementation of the core tree logic (Gini Impurity calculation and recursive splitting). |
| **`airports.csv` Use** | Compliant | The project successfully loads and processes the `airports.csv` file. |
| **PyScaffold Structure** | Compliant | The project structure separates code (`src/`), documentation (`docs/`), tests (`tests/`), and results (`experiments/`). |

## 3. Algorithm Implementation Details (`SimpleCART`)

The **`SimpleCART`** class defines the core decision tree logic.

### A. Splitting Criterion: Gini Impurity (What it Counts)
* **Definition:** Gini Impurity is a measure of the **homogeneity (purity)** of the class labels in a node.
* **Mechanism:** The model calculates the Gini Impurity before and after every potential split to determine the **Information Gain (Zysk)**. The split that provides the highest Gain (lowest resulting Gini Impurity) is chosen as the optimal decision rule for that node.
* **Goal:** The algorithm always chooses a split that makes the resulting child nodes **less mixed** (Gini close to 0).

### B. Core Splitting Logic and Optimization
* **Optimization (Threshold Sampling):** The method utilizes **Threshold Sampling** in its `_best_split` logic. For continuous numerical features, the model samples a maximum of **20 random unique values** per feature instead of testing all possible thresholds. This optimization is crucial to prevent the program from hanging (freezing) on the large, dense dataset.
* **Stopping Conditions:** The tree stops growing to prevent overfitting if:
    1.  The node is perfectly pure (Gini Impurity = 0.0).
    2.  The maximum depth (`max_depth=7`) is reached.
    3.  The node contains too few samples to justify splitting (`min_samples_split=5`).

## 4. Data Preprocessing

The preprocessing stage ensures the large dataset is prepared for efficient model training.

* **Row Sampling:** The dataset is intentionally limited to **3,000 random rows** (`sample_rows=3000`) for training to ensure the custom `SimpleCART` model trains within an acceptable timeframe.
* **Feature Engineering:** Features include numerical data (`latitude_deg`, `longitude_deg`, `elevation_ft`) and categorical data (`continent`, `iso_country`).
* **Categorical Feature Optimization:** The code limits the features used for **One-Hot Encoding (OHE)** by only tracking the Top 10 most frequent countries and grouping the rest into an 'OTHER' category, reducing the total number of features and increasing speed.
* **Data Cleaning:** Missing values (NaN) in numerical columns are filled with the **median**, and categorical values are filled with the string **'missing'**.
* **Encoding:** The final step converts all categorical data to numbers using OHE and the target variable (`type`) to integers using **LabelEncoder**.

## 5. Execution and Testing

### A. Project Structure
The project utilizes the following execution structure:
* **`src/main.py`:** Contains the entire `SimpleCART` class, preprocessing logic, training routine, and the final output comparison.

### B. How to Run `main.py`
To run the project, train both models, and see the final comparison output:

1.  **Activate Virtual Environment:** Ensure your terminal shows `(venv)`.
2.  **Execute:** Run the primary execution file from the project root:
    ```bash
    python src/main.py
    ```
3.  **Output:** The terminal displays the total number of samples loaded, the **Accuracy** for both the custom and Scikit-learn models, and their respective **Confusion Matrices**.

### C. Verification (Unit Tests)
The logic is verified through unit tests (in `tests/test_cart_logic.py`) which confirm the mathematical correctness of core functions, particularly the **`gini_impurity()`** calculation.
