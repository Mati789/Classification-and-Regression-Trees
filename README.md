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
python src/airport_random_forest/main.py
```

### D. Run the tests

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_basic.py
```

## Project Interview

This project implements a decision tree classifier from scratch using Gini Impurity as the splitting metric.

* Custom Implementation: SimpleCART class written purely in Python/NumPy — no reliance on sklearn for tree building.

* Performance Optimization: Implements a stochastic threshold sampling method to speed up training on continuous variables.

* Model Comparison: Automatically benchmarks the custom model against
sklearn.tree.DecisionTreeClassifier.

### Dataset

The project uses airports.csv (included in the root directory). The model predicts the airport type (small, medium, large) based on:

* Latitude & Longitude
* Elevation
* Continent & Country (One-Hot Encoded)


### Project structure

```bash
├── docs/
│   └── cart_documentation.md      # Detailed technical documentation of the algorithm
├── src/
│   ├── __init__.py
│   └── main.py                    # Main entry point: data loading, training, comparison
├── tests/
│   └── test_cart_logic.py         # Unit tests for Gini calculation and splits
├── airports.csv                   # Dataset
├── README.md                      # This file
└── requirements.txt               # Dependencies
```



