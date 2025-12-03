# CART Decision Tree Project Documentation

## 1. Project Requirements and Compliance

This project follows the official course requirements:
* **VirtualEnv:** Use of an isolated virtual environment for dependency management.
* **PyScaffold project structure:** Adherence to the project directory structure (`src/`, `tests/`, `docs/`, `experiments/`).
* **Own Implementation vs Scikit-learn:** Comparison of results between the custom implementation (`SimpleCART`) and the robust ready-made model (`DecisionTreeClassifier` from Scikit-learn).
* **Complete documentation:** Preparation of required technical documentation and reports.
* **Basic unit tests:** Requirement to prepare basic unit tests to verify core logic (e.g., Gini Impurity).

The project uses the data file **`airports.csv`**.

---

## 2. CART Algorithm Implementation (SimpleCART)

### A. What is it? (The Model)
`SimpleCART` is our custom implementation of a **single Decision Tree**. The model learns a series of questions (splits) based on the data to classify airport types.

### B. The Splitting Criterion (What it Counts)
The model calculates the best split using **Gini Impurity** (Niejednorodność Gini).
* **What it is:** Gini Impurity measures how **mixed** the classes (airport types) are within a given node.
* **Goal:** The algorithm chooses the split that **maximally reduces** the Gini Impurity, aiming for the **purest** possible child nodes (Gini close to 0.0). 

[Image of Gini Impurity formula]


### C. Operating Rules (Logic)
* **Building:** The tree recursively divides the data, seeking the best split for each node.
* **Optimization (Threshold Sampling):** Due to the large dataset size, the splitting logic only tests **20 random thresholds** per continuous feature, instead of every unique value, to ensure fast training.
* **Stopping:** The tree stops growing to prevent overfitting when:
    * The **maximum depth** (`max_depth=7`) is reached.
    * The node is perfectly pure (Gini = 0.0).
    * The node is too small to split (`min_samples_split=5`).

---

## 3. Data Preparation and Structure

### A. Data Preprocessing (Steps)
Data is prepared to ensure the model can learn efficiently:
* **Row Sampling:** Only **3,000 random rows** are loaded from `airports.csv` to ensure fast training times.
* **Feature Encoding:** Categorical features (e.g., Continent, Country) are converted to a numerical format (One-Hot Encoding).
* **Feature Optimization:** Only the **Top 10** most frequent countries are used as separate features; all others are grouped together to reduce the total number of features.
* **Handling Missing Values:** Missing data is automatically handled (e.g., elevation is filled with the median).

### B. Project Structure
All core logic (the `SimpleCART` class, preprocessing, and execution) is contained within the single file **`src/main.py`** for simplicity.

---

## 4. Execution and Verification

### A. How to Run `main.py`
To run the project, train both models, and display the final comparison:

1.  **Activate Virtual Environment:** Ensure your terminal shows `(venv)`.
2.  **Execute:** Run the primary execution file from the project root:
    ```bash
    python src/main.py
    ```

### B. Expected Output
The terminal will display:
1.  Information on the number of samples loaded for each class (e.g., small\_airport).
2.  The status of **MODEL TRAINING** and **Training FINISHED**.
3.  The final **Accuracy** for your custom `SimpleCART` model and the Scikit-learn model.
4.  Two **Confusion Matrices**, showing how accurately each model classified the test airport types.
