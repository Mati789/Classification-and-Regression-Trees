import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier 
import time
import sys
import warnings

# Suppress FutureWarnings from libraries for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(linewidth=200)

# ==============================================================================
# I. MODEL CLASS: SimpleCART (Your Custom Implementation)
# ==============================================================================

class SimpleCART:
    """
    Custom, simple implementation of the Classification and Regression Tree (CART) 
    for classification, using Gini Impurity as the splitting criterion.
    """
    def __init__(self, max_depth=7, min_samples_split=5):
        self.max_depth = max_depth          
        self.min_samples_split = min_samples_split
        self.tree = None
        
    @staticmethod
    def gini_impurity(y):
        """Calculates Gini Impurity (the measure the tree seeks to minimize)."""
        if len(y) == 0:
            return 0.0
        
        counts = Counter(y)
        n = len(y)
        impurity = 1.0
        
        for label in counts:
            p_i = counts[label] / n
            impurity -= p_i**2
        return impurity

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split the data (Gain maximization)."""
        best_gain = -1.0 
        best_idx = None
        best_thr = None

        n_samples, n_features = X.shape
        
        for idx in range(n_features):
            X_column = X[:, idx]
            
            # OPTIMIZATION: Threshold Sampling for Speed
            unique_values = np.unique(X_column)
            if len(unique_values) > 20:
                thresholds = np.random.choice(unique_values, 20, replace=False)
            else:
                thresholds = unique_values

            for thr in thresholds:
                left_indices = X_column <= thr
                right_indices = X_column > thr
                
                if np.sum(left_indices) < self.min_samples_split or np.sum(right_indices) < self.min_samples_split:
                    continue

                gini_before_split = self.gini_impurity(y)
                
                n_left = np.sum(left_indices)
                n_right = np.sum(right_indices)
                n_total = n_samples
                
                gini_left = self.gini_impurity(y[left_indices])
                gini_right = self.gini_impurity(y[right_indices])
                
                gini_after_split = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
                
                gain = gini_before_split - gini_after_split
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr, best_gain

    def _build_tree(self, X, y, depth=0):
        """Recursively builds the decision tree structure."""
        
        if self.gini_impurity(y) == 0.0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        
        idx, thr, gain = self._best_split(X, y)

        if idx is None or gain <= 0:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}

        X_column = X[:, idx]
        left_indices = X_column <= thr
        right_indices = X_column > thr
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'leaf': False, 
            'feature_idx': idx, 
            'threshold': thr, 
            'left': left_subtree, 
            'right': right_subtree
        }

    def fit(self, X, y):
        """Trains the model by building the decision tree."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.tree = self._build_tree(X, y)
        
    def _predict_single(self, row, node):
        """Recursively traverses the tree for a single sample."""
        if node['leaf']:
            return node['value']
        
        feature_value = row[node['feature_idx']]

        if feature_value <= node['threshold']:
            return self._predict_single(row, node['left'])
        else:
            return self._predict_single(row, node['right'])
            
    def predict(self, X):
        """Predicts labels for the entire dataset X."""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        predictions = [self._predict_single(row, self.tree) for row in X]
        return np.array(predictions)


# ==============================================================================
# II. DATA PREPROCESSING (Optimized)
# ==============================================================================

def load_and_preprocess_data(path='airports.csv', test_size=0.2, random_state=42, sample_rows=3000):
    """
    Loads data, samples 3000 rows for faster training, and prepares features.
    """
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: Data file '{path}' not found.")
        sys.exit(1)
        
    # STEP 1: ROW SAMPLING (Limit dataset size)
    if sample_rows and len(data) > sample_rows:
        data = data.sample(n=sample_rows, random_state=random_state).copy()
    
    # Feature Selection and Optimization
    FEATURES = ['latitude_deg', 'longitude_deg', 'elevation_ft', 'continent']
    
    # Aggressively limit OHE: Add top 10 most frequent countries
    top_countries = data['iso_country'].value_counts().nlargest(10).index.tolist()
    data['iso_country_top'] = data['iso_country'].apply(lambda x: x if x in top_countries else 'OTHER')
    FEATURES.append('iso_country_top')
    
    TARGET = 'type'
    relevant_types = ['small_airport', 'medium_airport', 'large_airport']
    data = data[data[TARGET].isin(relevant_types)].copy()
    
    # Handle NaN 
    data['elevation_ft'] = data['elevation_ft'].fillna(data['elevation_ft'].median())
    data['continent'] = data['continent'].fillna('missing')
    data['iso_country_top'] = data['iso_country_top'].fillna('missing')


    # One-Hot Encoding
    X = pd.get_dummies(data[FEATURES], columns=['continent', 'iso_country_top'], drop_first=True)
    y = data[TARGET]

    # Label Encoding (Target conversion to 0, 1, 2...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-Test Split (Stratified to maintain class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Get class counts for output
    class_counts = pd.Series(y_train).value_counts().sort_index().to_dict()
    class_names = le.classes_
    
    # 1. Outputting Data Info (REQUESTED FORMAT)
    print("=" * 60)
    print(f"SAMPLES LOADED FOR TRAINING AND TESTING: {len(X_train) + len(X_test)} samples")
    for encoded_label, count in class_counts.items():
        print(f"   {encoded_label}. {class_names[encoded_label]}: {count} samples")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, le


# ==============================================================================
# III. MAIN PROJECT EXECUTION AND EVALUATION
# ==============================================================================

def run_cart_project():
    """Main function to run the entire process: load, train, and evaluate."""
    
    # 1. Load and Preprocess (sampling 3000 rows)
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(sample_rows=3000)
    target_names = label_encoder.classes_

    # Model Settings (Same for both)
    MAX_DEPTH = 7
    MIN_SAMPLES = 5
    
    # ----------------------------------------------------------------------
    # 2A & 2B. Trenowanie Modeli
    # ----------------------------------------------------------------------
    print(f"2. MODEL TRAINING:")
    
    # WŁASNY MODEL
    print("   => Training CUSTOM SimpleCART...")
    cart_model = SimpleCART(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES)
    cart_model.fit(X_train, y_train)
    y_pred_custom = cart_model.predict(X_test)

    # SCIKIT-LEARN MODEL
    print("   => Training Scikit-learn (DecisionTreeClassifier)...")
    sklearn_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=MAX_DEPTH, 
        min_samples_split=MIN_SAMPLES,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    print("   Training FINISHED.") # Wymagane
    
    # ----------------------------------------------------------------------
    # 3. PORÓWNANIE WYNIKÓW
    # ----------------------------------------------------------------------
    print("\n" + "=" * 25 + " RESULTS COMPARISON " + "=" * 25)
    
    # METRYKI WŁASNE
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)

    # METRYKI SCIKIT-LEARN
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    
    # 3. Wypisanie Accuracy
    print(f"\nACCURACY:")
    print(f"  CUSTOM SimpleCART: {accuracy_custom:.4f}")
    print(f"  SCIKIT-LEARN:      {accuracy_sklearn:.4f}")

    # 4. Macierze Pomyłek
    print("\n" + "=" * 10 + " CONFUSION MATRIX (CUSTOM MODEL) " + "=" * 10)
    print(pd.DataFrame(conf_matrix_custom, index=target_names, columns=target_names))
    
    print("\n" + "=" * 10 + " CONFUSION MATRIX (SCIKIT-LEARN) " + "=" * 10)
    print(pd.DataFrame(conf_matrix_sklearn, index=target_names, columns=target_names))
    
    print("\n" + "=" * 60)
    
# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    run_cart_project()