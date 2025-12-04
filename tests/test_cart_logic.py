"""Test suite for CART project"""
import pytest
import pandas as pd
import numpy as np
import sys, os, tempfile
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import load_and_preprocess_data, calculate_entropy, calculate_information_gain, DATASET_FILE, ROWS_LIMIT

class TestDataLoading:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_data("nonexistent.csv", 100)
    
    def test_successful_loading(self):
        X, y, features, le = load_and_preprocess_data(DATASET_FILE, 500)
        assert len(X) > 0 and len(X) == len(y) and len(X) <= 500
        assert features == ['Airline', 'Source', 'Destination']
        assert isinstance(le, LabelEncoder)
    
    def test_no_missing_values(self):
        X, y, features, le = load_and_preprocess_data(DATASET_FILE, 300)
        assert not X.isnull().any().any() and not y.isnull().any()
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

class TestEntropyCalculation:
    def test_entropy_edge_cases(self):
        assert calculate_entropy(pd.Series([])) == 0.0
        assert calculate_entropy(pd.Series([1, 1, 1])) < 0.01
        assert abs(calculate_entropy(pd.Series([1]))) < 1e-6
    
    def test_entropy_balanced(self):
        entropy = calculate_entropy(pd.Series([0, 0, 0, 0, 1, 1, 1, 1]))
        assert 0.99 < entropy <= 1.01
    
    def test_entropy_multi_class(self):
        entropy = calculate_entropy(pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2]))
        assert abs(entropy - np.log2(3)) < 0.01
    
    def test_entropy_real_data(self):
        X, y, _, _ = load_and_preprocess_data(DATASET_FILE, 1000)
        entropy = calculate_entropy(y)
        assert 0 <= entropy <= 10 and isinstance(entropy, float)

class TestInformationGain:
    def test_perfect_split(self):
        X = pd.DataFrame({'f': [0, 0, 0, 1, 1, 1]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        assert calculate_information_gain(X, y, 'f') > 0.9
    
    def test_no_split(self):
        X = pd.DataFrame({'f': [0, 0, 0, 0, 0, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        assert abs(calculate_information_gain(X, y, 'f')) < 0.01
    
    def test_feature_comparison(self):
        X = pd.DataFrame({'good': [0, 0, 0, 1, 1, 1], 'bad': [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        assert calculate_information_gain(X, y, 'good') > calculate_information_gain(X, y, 'bad')
    
    def test_real_data_all_features(self):
        X, y, features, _ = load_and_preprocess_data(DATASET_FILE, 500)
        ig_values = {f: calculate_information_gain(X, y, f) for f in features}
        assert len(ig_values) == 3 and all(ig >= 0 for ig in ig_values.values())

class TestModelTraining:
    def test_simple_model(self):
        X, y, _, _ = load_and_preprocess_data(DATASET_FILE, 800)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X, y)
        assert hasattr(model, 'tree_') and model.get_depth() <= 3
    
    def test_model_predictions(self):
        X, y, _, _ = load_and_preprocess_data(DATASET_FILE, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test) and set(preds).issubset(set(y.unique()))
    
    def test_model_accuracy(self):
        X, y, _, _ = load_and_preprocess_data(DATASET_FILE, 1500)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 1.0 / len(y.unique()) and 0 <= acc <= 1

class TestIntegration:
    def test_complete_workflow(self):
        X, y, features, _ = load_and_preprocess_data(DATASET_FILE, 800)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        entropy = calculate_entropy(y_train)
        ig_values = {f: calculate_information_gain(X_train, y_train, f) for f in features}
        best = max(ig_values, key=ig_values.get)
        
        assert entropy >= 0 and best in features
        
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        assert 0 <= accuracy_score(y_test, model.predict(X_test)) <= 1
    
    def test_model_comparison(self):
        X, y, _, _ = load_and_preprocess_data(DATASET_FILE, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        
        model1 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model2 = DecisionTreeClassifier(criterion='gini', random_state=42)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        acc1 = accuracy_score(y_test, model1.predict(X_test))
        acc2 = accuracy_score(y_test, model2.predict(X_test))
        assert 0 <= acc1 <= 1 and 0 <= acc2 <= 1 and model2.get_depth() > 3

class TestConstants:
    def test_values(self):
        assert DATASET_FILE == "dataset.csv" and ROWS_LIMIT == 5000 and isinstance(ROWS_LIMIT, int)

