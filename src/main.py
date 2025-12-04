import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import random 
from typing import Tuple

# Global constants
DATASET_FILE = "dataset.csv"
ROWS_LIMIT = 5000

def load_and_preprocess_data(file_path: str, limit: int) -> Tuple[pd.DataFrame, pd.Series, list, LabelEncoder]:
    """
    Loads data from a CSV file, limits rows, selects features and target,
    and performs Label Encoding on all columns.
    
    Returns: X (features), y (target), list_of_features, LabelEncoder for target (y).
    """
    # --- 1. Data Loading, Limiting, and Cleaning ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Ensure it is in the same directory.")
        raise

    # Limit to the first N rows
    df = df.head(limit)

    # Select columns: 3 features (X) and 1 target column (Y)
    features = ['Airline', 'Source', 'Destination']
    target = 'Total_Stops'

    # Select the subset of data
    data = df[features + [target]].copy()

    # Data Cleaning: Drop rows with missing values in the selected columns (simplest approach)
    data.dropna(subset=features + [target], inplace=True)

    # --- 2. Preprocessing: Label Encoding ---
    # Convert categorical text labels into numerical values
    le_target = LabelEncoder()
    
    # Encoding features (X) and target (y)
    for col in features:
        # Using a separate Label Encoder for each feature column for simplicity
        data[col] = LabelEncoder().fit_transform(data[col]) 
        
    # Encoding target (y)
    data[target] = le_target.fit_transform(data[target])

    # Prepare data for modeling
    X = data[features]
    y = data[target]

    return X, y, features, le_target

def calculate_entropy(y_data: pd.Series) -> float:
    """Calculates Entropy for a given set of labels (y)."""
    if len(y_data) == 0:
        return 0.0
    probabilities = y_data.value_counts(normalize=True)
    # Entropy = -sum(p * log2(p))
    # Add 1e-9 to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9)) 
    return entropy

def calculate_information_gain(X_data: pd.DataFrame, y_data: pd.Series, feature_col: str) -> float:
    """Calculates Information Gain after splitting based on a given feature column."""
    # Parent entropy (of the whole set)
    entropy_parent = calculate_entropy(y_data)
    
    # Weighted child entropy
    weighted_entropy_children = 0.0
    total_samples = len(y_data)
    
    # Iterate through unique feature values (splits)
    for value in X_data[feature_col].unique():
        y_subset = y_data[X_data[feature_col] == value]
        weight = len(y_subset) / total_samples
        weighted_entropy_children += weight * calculate_entropy(y_subset)
        
    # Information Gain = Parent_Entropy - Weighted_Child_Entropy
    information_gain = entropy_parent - weighted_entropy_children
    return information_gain

def main():
    """Main function to run the entire CART analysis process."""
    # Generate a random seed for variability across multiple script executions.
    random_seed = random.randint(0, 100000)
    
    print(f"--- STARTING CART PROJECT (Entropy) ---")
    print(f"Using random state seed: {random_seed}. Results will vary on each run.")
    
    # 1. Load and Preprocess Data
    try:
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, ROWS_LIMIT)
    except Exception as e:
        print(f"Script terminated due to data loading error: {e}")
        return

    # Split into training and testing sets (30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    print(f"Successfully loaded and split data ({len(X_train)} training samples, {len(X_test)} test samples).")
    
    # --- 2. Entropy and Information Gain (IG) Demonstration ---
    print("\n--- ENTROPY AND INFORMATION GAIN DEMONSTRATION ---")
    print(f"Training set entropy: {calculate_entropy(y_train):.4f}")
    print("Information Gain for features (higher value = better split):")
    
    best_ig = -1
    best_feature = ""
    
    for feature in features:
        ig = calculate_information_gain(X_train, y_train, feature)
        print(f"  - {feature}: {ig:.4f}")
        if ig > best_ig:
            best_ig = ig
            best_feature = feature
            
    print(f"Conclusion: The best feature for the first split is '{best_feature}' (IG: {best_ig:.4f}).")

    # --- 3. Modeling and Comparison ---

    # 1. SIMPLE CART MODEL: Entropy criterion, increased max_depth=10 for better Accuracy
    simple_cart_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=random_seed)
    simple_cart_model.fit(X_train, y_train)
    y_pred_simple = simple_cart_model.predict(X_test)

    # 2. SCIKIT-LEARN COMPARISON MODEL: Default Gini criterion, full depth
    sklearn_model = DecisionTreeClassifier(criterion='gini', random_state=random_seed)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    # --- 4. Evaluation and Results ---

    # Results for the Simple CART Model
    accuracy_simple = accuracy_score(y_test, y_pred_simple)
    cm_simple = confusion_matrix(y_test, y_pred_simple)

    # Results for the Scikit-learn Comparison Model
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    
    print("\n--- MODELING RESULTS AND EVALUATION ---")

    print("\n1. SIMPLE CART MODEL (Entropy, max_depth=10 - IMPROVED)")
    print(f"  - Accuracy: {accuracy_simple:.4f}")
    print("  - Confusion Matrix:")
    print(cm_simple)

    print("\n2. SCIKIT-LEARN COMPARISON MODEL (Gini, full depth)")
    print(f"  - Accuracy: {accuracy_sklearn:.4f}")
    print("  - Confusion Matrix:")
    print(cm_sklearn)
    
    print("\n--- PROJECT END ---")

if __name__ == "__main__":
    main()
