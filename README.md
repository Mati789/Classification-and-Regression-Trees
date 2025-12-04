# CART Algorithm Implementation

This repository contains a custom implementation of the **Classification and Regression Trees (CART)** algorithm, developed as part of a Python Programming project. The model classifies airports based on geographical features and is benchmarked against `scikit-learn`.

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You will also need the following libraries:
* numpy
* pandas
* scikit-learn
* pytest (for running tests)

## How to Run

### A. Create and activate VirtualEnv

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### B. Install dependencies

```bash
pip install -r requirements.txt
```

### C. Run the main file

```bash
python src/main.py
```

### D. Run the tests

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with verbose output
pytest -v

```

## 💾 Dataset and Data Utilization

The project uses the **'dataset.csv'** file, which contains information about flight ticket prices, routes, airlines, and total stopovers.

### Data Utilization Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Rows Used** | 5000 | Only the first 5000 rows of the dataset are utilized for training and testing. |
| **Features (X)** | `Airline`, `Source`, `Destination` (3 columns) | These categorical columns are used as input for the models. |
| **Target (Y)** | `Total_Stops`  | This is the categorical variable the models are trained to predict. |
| **Train/Test Split** | 90% Training / 10% Testing | The data is split into a large training set (approx. 4500 rows) and a smaller testing set (approx. 500 rows) for model validation. |

---

## 🌳 Classification Algorithm: CART with Entropy

The core of this project is the **Classification and Regression Tree (CART)** algorithm, specifically implemented as a **Decision Tree Classifier**.

### Why Entropy?

The model utilizes **Entropy** as the primary criterion for splitting nodes in the tree.

1.  **Entropy Calculation:** Entropy measures the **impurity** or **randomness** of the class distribution in a dataset. A low entropy value (close to 0) means the data is homogeneous (pure).
2.  **Information Gain (IG) 


:** The model calculates the **Information Gain**—the reduction in entropy—achieved by splitting the data using a specific feature. The split that yields the **highest IG** is chosen, as it provides the most useful information for classification.

### Model Comparison

The project compares two models trained on the same data split:

1.  **Simple CART Model :** Uses the **Entropy** criterion but is **limited to a maximum depth of 3** (`max_depth=3`). This model serves as a simple baseline.
2.  **Scikit-learn Comparison Model:** Uses the default **Gini** criterion and is allowed to grow to its full depth (`max_depth=None`). This model acts as a reference for maximum achievable accuracy.

---

## 📂 Project Structure

```
project_root/
├── README.md
├── dataset.csv             # Kaggle airport dataset
├── docs/
│   └── cart_documentation.md     # description of algorithm + how it works
├── src/
│   └── main.py              #  data loading, training, comparison
├── tests/
│   └── test_cart_logic.py        #  tests 
├── requirements.txt
└── setup.cfg                # PyScaffold configuration
```
