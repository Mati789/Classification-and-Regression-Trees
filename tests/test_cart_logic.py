"""
Comprehensive test suite for CART (Classification and Regression Trees) project.
Tests all core functionality, data processing, calculations, and model training.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from unittest.mock import patch, MagicMock
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from main import (
    load_and_preprocess_data,
    calculate_entropy,
    calculate_information_gain,
    DATASET_FILE,
    ROWS_LIMIT
)


class TestDataLoading:
    """Tests for data loading and preprocessing functionality."""
    
    def test_load_and_preprocess_data_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_data("nonexistent_file.csv", 100)
    
    def test_load_and_preprocess_data_success(self):
        """Test successful data loading and preprocessing."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        # Check that data is loaded
        assert len(X) > 0, "X should not be empty"
        assert len(y) > 0, "y should not be empty"
        assert len(X) == len(y), "X and y should have same length"
        
        # Check features
        assert features == ['Airline', 'Source', 'Destination']
        assert list(X.columns) == features
        
        # Check that LabelEncoder is returned
        assert isinstance(le_target, LabelEncoder)
        
        # Check that data is numerical (encoded)
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        assert np.issubdtype(y.dtype, np.number)
    
    def test_load_and_preprocess_data_limit(self):
        """Test that row limit is applied correctly."""
        limit = 500
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, limit)
        
        # Should be at most 'limit' rows (could be less if NA values dropped)
        assert len(X) <= limit
        assert len(y) <= limit
    
    def test_load_and_preprocess_data_no_missing_values(self):
        """Test that missing values are properly removed."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        # No NaN values should remain
        assert not X.isnull().any().any(), "X should not contain NaN values"
        assert not y.isnull().any(), "y should not contain NaN values"
    
    def test_load_and_preprocess_with_custom_csv(self):
        """Test data loading with custom CSV file."""
        # Create a temporary CSV file
        temp_data = pd.DataFrame({
            'Airline': ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet'],
            'Source': ['Delhi', 'Mumbai', 'Kolkata', 'Banglore'],
            'Destination': ['Mumbai', 'Delhi', 'Banglore', 'Delhi'],
            'Total_Stops': ['non-stop', '1 stop', '2 stops', 'non-stop']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X, y, features, le_target = load_and_preprocess_data(temp_file, 10)
            assert len(X) == 4
            assert len(y) == 4
            assert list(X.columns) == ['Airline', 'Source', 'Destination']
        finally:
            os.unlink(temp_file)
    
    def test_encoded_values_are_integers(self):
        """Test that encoded values are integers."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 100)
        
        # All values should be integers
        for col in X.columns:
            assert X[col].dtype in [np.int32, np.int64], f"{col} should be integer type"
        assert y.dtype in [np.int32, np.int64], "y should be integer type"


class TestEntropyCalculation:
    """Tests for entropy calculation functionality."""
    
    def test_entropy_empty_series(self):
        """Test entropy of empty series."""
        y = pd.Series([])
        entropy = calculate_entropy(y)
        assert entropy == 0.0
    
    def test_entropy_pure_series(self):
        """Test entropy of pure (single class) series."""
        y = pd.Series([1, 1, 1, 1, 1])
        entropy = calculate_entropy(y)
        # Pure series should have entropy close to 0
        assert entropy < 0.01
    
    def test_entropy_balanced_binary(self):
        """Test entropy of balanced binary distribution."""
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        entropy = calculate_entropy(y)
        # Balanced binary should have entropy close to 1
        assert 0.99 < entropy <= 1.01
    
    def test_entropy_multi_class(self):
        """Test entropy with multiple classes."""
        y = pd.Series([0, 0, 1, 1, 2, 2])
        entropy = calculate_entropy(y)
        # Should be positive and reasonable
        assert 0 < entropy < 2
    
    def test_entropy_unbalanced(self):
        """Test entropy of unbalanced distribution."""
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        entropy = calculate_entropy(y)
        # Unbalanced should have lower entropy than balanced
        assert 0 < entropy < 1
    
    def test_entropy_with_real_data(self):
        """Test entropy calculation with real dataset."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        entropy = calculate_entropy(y)
        
        # Entropy should be positive and reasonable
        assert 0 <= entropy <= 10  # Upper bound depends on number of classes
        assert isinstance(entropy, float)
    
    def test_entropy_three_equal_classes(self):
        """Test entropy with three equally distributed classes."""
        y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])
        entropy = calculate_entropy(y)
        # log2(3) ≈ 1.585
        expected = np.log2(3)
        assert abs(entropy - expected) < 0.01


class TestInformationGain:
    """Tests for information gain calculation functionality."""
    
    def test_information_gain_perfect_split(self):
        """Test information gain with perfect split."""
        X = pd.DataFrame({
            'feature': [0, 0, 0, 1, 1, 1]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        ig = calculate_information_gain(X, y, 'feature')
        # Perfect split should have high IG
        assert ig > 0.9
    
    def test_information_gain_no_split(self):
        """Test information gain with no meaningful split."""
        X = pd.DataFrame({
            'feature': [0, 0, 0, 0, 0, 0]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        ig = calculate_information_gain(X, y, 'feature')
        # No split should have IG close to 0
        assert abs(ig) < 0.01
    
    def test_information_gain_with_real_data(self):
        """Test information gain with real dataset."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        for feature in features:
            ig = calculate_information_gain(X, y, feature)
            # IG should be non-negative
            assert ig >= 0, f"IG for {feature} should be non-negative"
            assert isinstance(ig, float)
    
    def test_information_gain_comparison(self):
        """Test that information gain can distinguish between features."""
        X = pd.DataFrame({
            'good_feature': [0, 0, 0, 1, 1, 1],
            'bad_feature': [0, 1, 0, 1, 0, 1]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        ig_good = calculate_information_gain(X, y, 'good_feature')
        ig_bad = calculate_information_gain(X, y, 'bad_feature')
        
        # Good feature should have higher IG
        assert ig_good > ig_bad
    
    def test_information_gain_all_features(self):
        """Test information gain for all features in dataset."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 500)
        
        ig_values = {}
        for feature in features:
            ig_values[feature] = calculate_information_gain(X, y, feature)
        
        # All IG values should be computed
        assert len(ig_values) == 3
        # At least one feature should have positive IG
        assert max(ig_values.values()) > 0


class TestModelTraining:
    """Tests for model training and predictions."""
    
    def test_simple_cart_model_creation(self):
        """Test that simple CART model can be created and trained."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Model should be fitted
        assert hasattr(model, 'tree_')
        assert model.tree_.node_count > 0
        
        # Model should have max_depth <= 3
        assert model.get_depth() <= 3
    
    def test_sklearn_model_creation(self):
        """Test that sklearn comparison model can be created and trained."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        model = DecisionTreeClassifier(criterion='gini', random_state=42)
        model.fit(X, y)
        
        # Model should be fitted
        assert hasattr(model, 'tree_')
        assert model.tree_.node_count > 0
        
        # Model should have depth > 3 (unrestricted)
        assert model.get_depth() > 3
    
    def test_model_predictions(self):
        """Test that models can make predictions."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Predictions should have same length as input
        assert len(predictions) == len(X)
        
        # Predictions should be in the range of y values
        assert set(predictions).issubset(set(y.unique()))
    
    def test_model_accuracy_range(self):
        """Test that model accuracy is within reasonable range."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Accuracy should be reasonable (better than random guessing)
        num_classes = len(y.unique())
        random_accuracy = 1.0 / num_classes
        assert accuracy > random_accuracy
        assert 0 <= accuracy <= 1


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete CART workflow from loading to prediction."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        # Load data
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # Calculate entropy
        entropy = calculate_entropy(y_train)
        assert entropy >= 0
        
        # Calculate information gain for all features
        ig_values = {}
        for feature in features:
            ig_values[feature] = calculate_information_gain(X_train, y_train, feature)
        
        # Find best feature
        best_feature = max(ig_values, key=ig_values.get)
        assert best_feature in features
        
        # Train models
        simple_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        simple_model.fit(X_train, y_train)
        
        sklearn_model = DecisionTreeClassifier(criterion='gini', random_state=42)
        sklearn_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_simple = simple_model.predict(X_test)
        y_pred_sklearn = sklearn_model.predict(X_test)
        
        # Calculate metrics
        acc_simple = accuracy_score(y_test, y_pred_simple)
        acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
        cm_simple = confusion_matrix(y_test, y_pred_simple)
        cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
        
        # Verify results
        assert 0 <= acc_simple <= 1
        assert 0 <= acc_sklearn <= 1
        assert cm_simple.shape[0] == cm_simple.shape[1]
        assert cm_sklearn.shape[0] == cm_sklearn.shape[1]
    
    def test_consistency_with_random_seed(self):
        """Test that results are consistent with same random seed."""
        from sklearn.model_selection import train_test_split
        
        seed = 12345
        
        # First run
        X1, y1, features1, le1 = load_and_preprocess_data(DATASET_FILE, 1000)
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.1, random_state=seed)
        model1 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=seed)
        model1.fit(X_train1, y_train1)
        pred1 = model1.predict(X_test1)
        
        # Second run
        X2, y2, features2, le2 = load_and_preprocess_data(DATASET_FILE, 1000)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.1, random_state=seed)
        model2 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=seed)
        model2.fit(X_train2, y_train2)
        pred2 = model2.predict(X_test2)
        
        # Results should be identical
        assert np.array_equal(pred1, pred2)
    
    def test_different_data_sizes(self):
        """Test that the workflow works with different data sizes."""
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, size)
            
            # Data should be loaded
            assert len(X) > 0
            assert len(y) > 0
            
            # Entropy should be calculable
            entropy = calculate_entropy(y)
            assert entropy >= 0
            
            # Information gain should be calculable
            for feature in features:
                ig = calculate_information_gain(X, y, feature)
                assert ig >= 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_row_entropy(self):
        """Test entropy with single row."""
        y = pd.Series([1])
        entropy = calculate_entropy(y)
        assert abs(entropy) < 1e-6  # Close to 0 with numerical precision tolerance
    
    def test_single_class_information_gain(self):
        """Test information gain when target has single class."""
        X = pd.DataFrame({'feature': [0, 1, 0, 1]})
        y = pd.Series([0, 0, 0, 0])
        
        ig = calculate_information_gain(X, y, 'feature')
        assert ig == 0.0
    
    def test_many_classes(self):
        """Test with many unique classes."""
        y = pd.Series(range(100))
        entropy = calculate_entropy(y)
        
        # Max entropy for 100 classes is log2(100)
        max_entropy = np.log2(100)
        assert abs(entropy - max_entropy) < 0.01
    
    def test_feature_with_many_unique_values(self):
        """Test information gain with feature having many unique values."""
        X = pd.DataFrame({'feature': range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        
        ig = calculate_information_gain(X, y, 'feature')
        assert ig >= 0


class TestConstants:
    """Tests for module constants."""
    
    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert DATASET_FILE is not None
        assert ROWS_LIMIT is not None
    
    def test_constants_values(self):
        """Test that constants have expected values."""
        assert DATASET_FILE == "dataset.csv"
        assert ROWS_LIMIT == 5000
        assert isinstance(ROWS_LIMIT, int)
        assert ROWS_LIMIT > 0


class TestDataTypes:
    """Tests for data types and structure."""
    
    def test_return_types(self):
        """Test that functions return correct types."""
        X, y, features, le_target = load_and_preprocess_data(DATASET_FILE, 100)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(features, list)
        assert isinstance(le_target, LabelEncoder)
    
    def test_entropy_return_type(self):
        """Test that entropy returns float."""
        y = pd.Series([0, 1, 0, 1])
        entropy = calculate_entropy(y)
        assert isinstance(entropy, (float, np.floating))
    
    def test_information_gain_return_type(self):
        """Test that information gain returns float."""
        X = pd.DataFrame({'f': [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        ig = calculate_information_gain(X, y, 'f')
        assert isinstance(ig, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
