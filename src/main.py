import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import sys

# Ustawienie opcji wyświetlania numpy, aby macierze były czytelne
np.set_printoptions(linewidth=200)

# ==============================================================================
# I. KLASA MODELU: SimpleCART (Decision Tree Implementation)
# ==============================================================================

class SimpleCART:
    """
    Własna, prosta implementacja Drzewa Decyzyjnego (CART) do klasyfikacji,
    wykorzystująca Gini Impurity jako kryterium podziału.
    """
    def __init__(self, max_depth=7, min_samples_split=5):
        # Ustawienie hiperparametrów
        self.max_depth = max_depth          # Maksymalna głębokość drzewa (reguła zatrzymania)
        self.min_samples_split = min_samples_split # Min. liczba próbek do podziału
        self.tree = None
        
    @staticmethod
    def gini_impurity(y):
        """Oblicza Gini Impurity (Niejednorodność Gini) dla zbioru etykiet y."""
        if len(y) == 0:
            return 0.0
        
        # Obliczanie proporcji (p_i) każdej klasy
        counts = Counter(y)
        n = len(y)
        impurity = 1.0
        
        # Wzór Gini: 1 - Suma(p_i^2)
        for label in counts:
            p_i = counts[label] / n
            impurity -= p_i**2
        return impurity

    def _best_split(self, X, y):
        """Znajduje najlepszą cechę i próg podziału poprzez maksymalizację Zysku (Gain)."""
        best_gain = -1.0 
        best_idx = None
        best_thr = None

        n_features = X.shape[1]
        
        # Iteracja po każdej kolumnie (cecha lotniska)
        for idx in range(n_features):
            X_column = X.iloc[:, idx]
            thresholds = np.unique(X_column)
            
            # Iteracja po każdym unikalnym progu w danej kolumnie
            for thr in thresholds:
                left_indices = X_column <= thr
                right_indices = X_column > thr
                
                # Reguła zatrzymania: Podział musi być wystarczająco duży
                if sum(left_indices) < self.min_samples_split or sum(right_indices) < self.min_samples_split:
                    continue

                # Obliczanie Gini po podziale (ważone)
                gini_before_split = self.gini_impurity(y)
                
                n_left = len(y[left_indices])
                n_right = len(y[right_indices])
                n_total = len(y)
                
                gini_left = self.gini_impurity(y[left_indices])
                gini_right = self.gini_impurity(y[right_indices])
                
                gini_after_split = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
                
                # Obliczanie Zysku (Gain) - to jest to, co chcemy maksymalizować
                gain = gini_before_split - gini_after_split
                
                # Aktualizacja najlepszego podziału
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr, best_gain

    def _build_tree(self, X, y, depth=0):
        """Rekurencyjnie buduje drzewo, używając najlepszych podziałów."""
        
        # Reguły zatrzymania (Stopping Conditions)
        # 1. Węzeł jest czysty, 2. osiągnięto max_depth, 3. zbyt mało próbek
        if self.gini_impurity(y) == 0.0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            # Tworzenie liścia (Leaf Node)
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        
        # Szukanie najlepszego podziału
        idx, thr, gain = self._best_split(X, y)

        # Jeśli nie znaleziono podziału o dodatnim zysku
        if idx is None or gain <= 0:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}

        # Przygotowanie danych do dalszej rekurencji
        X_column = X.iloc[:, idx]
        left_indices = X_column <= thr
        right_indices = X_column > thr
        
        # Rekurencyjne budowanie lewego i prawego poddrzewa
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Zwracanie węzła wewnętrznego (Node)
        return {
            'leaf': False, 
            'feature_idx': idx, 
            'threshold': thr, 
            'left': left_subtree, 
            'right': right_subtree
        }

    def fit(self, X, y):
        """Trenuje model budując drzewo decyzyjne."""
        # Resetowanie indeksów zapobiega problemom po podziale danych
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.tree = self._build_tree(X, y)
        
    def _predict_single(self, row, node):
        """Rekurencyjnie przechodzi przez drzewo dla pojedynczej próbki (predykcja)."""
        # Jeśli dotarliśmy do liścia, zwracamy klasę
        if node['leaf']:
            return node['value']
        
        # Pobieramy wartość cechy dla bieżącego podziału
        feature_value = row[node['feature_idx']]

        # Podejmujemy decyzję i schodzimy w dół
        if feature_value <= node['threshold']:
            return self._predict_single(row, node['left'])
        else:
            return self._predict_single(row, node['right'])
            
    def predict(self, X):
        """Przewiduje etykiety dla całego zbioru X."""
        # Konwertujemy DataFrame na NumPy dla szybkiego indeksowania podczas przewidywania
        X_np = X.values
        # Używamy list comprehension do przewidywania dla wszystkich próbek
        predictions = [self._predict_single(row, self.tree) for row in X_np]
        return np.array(predictions)


# ==============================================================================
# II. PREPROCESSING DANYCH
# ==============================================================================

def load_and_preprocess_data(path='airports.csv', test_size=0.2, random_state=42):
    """
    Ładuje dane, wykonuje Feature Engineering, obsługuje braki danych i podział na zbiory.
    """
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: Plik danych '{path}' nie został znaleziony. Upewnij się, że jest w głównym katalogu.")
        sys.exit(1)
    
    # Wybrane Cechy (rozszerzone o kategoryczne, aby dać modelowi szansę na naukę)
    FEATURES = ['latitude_deg', 'longitude_deg', 'elevation_ft', 'iso_country', 'continent']
    TARGET = 'type'
    
    # 1. Filtracja (Ograniczamy się do typów lotnisk, które chcemy klasyfikować)
    relevant_types = ['small_airport', 'medium_airport', 'large_airport']
    data = data[data[TARGET].isin(relevant_types)].copy()
    
    # 2. Obsługa NaN (Numeryczne medianą, Kategoryczne 'missing')
    for col in ['elevation_ft']:
        data[col].fillna(data[col].median(), inplace=True)
    
    for col in ['iso_country', 'continent']:
        data[col].fillna('missing', inplace=True)

    # 3. One-Hot Encoding (Konwersja cech kategorycznych na kolumny binarne)
    X = pd.get_dummies(data[FEATURES], columns=['iso_country', 'continent'], drop_first=True)
    y = data[TARGET]

    # 4. Kodowanie Etykiet (Zamiana stringów na liczby 0, 1, 2...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. Podział na Zbiory Treningowy i Testowy (Stratified, aby utrzymać proporcje klas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Informacja o niezrównoważeniu klas
    print(f"Rozkład klas po preprocessingu (Trening): {pd.Series(y_train).value_counts()}")
    
    # UWAGA: W pełni profesjonalnym projekcie, tutaj powinno być balansowanie SMOTE.
    # Ze względu na prostotę i brak zewnętrznego importu, pomijamy ten krok.
    
    return X_train, X_test, y_train, y_test, le

# ==============================================================================
# III. GŁÓWNA FUNKCJA PROJEKTU I OCENA
# ==============================================================================

def run_cart_project():
    """Główna funkcja uruchamiająca cały proces: ładowanie, trenowanie i ocenę."""
    print("=" * 60)
    print("--- STARTING CART PROJECT: CLASSIFICATION AND REGRESSION TREES ---")
    print("=" * 60)
    
    # 1. Ładowanie i Preprocessing
    print("\n1. Ładowanie i Preprocessing Danych...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()

    # 2. Model Training
    MAX_DEPTH = 7
    MIN_SAMPLES = 5
    print(f"\n2. Trenowanie modelu SimpleCART (Max Depth: {MAX_DEPTH}, Min Samples: {MIN_SAMPLES})...")
    
    # Inicjalizacja i trenowanie
    cart_model = SimpleCART(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES)
    start_time = time.time()
    cart_model.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"   Model SimpleCART wytrenowany w {end_time - start_time:.2f} sekundy.")

    # 3. Model Evaluation
    print("\n3. Ocena modelu na zbiorze testowym...")
    y_pred = cart_model.predict(X_test)
    
    # Odwracanie kodowania dla czytelności raportu
    target_names = label_encoder.classes_
    
    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 4. Wyświetlanie Wyników (Raport)
    print("\n" + "=" * 25 + " FINAL RESULTS " + "=" * 25)
    print(f"Overall Accuracy (Dokładność Ogólna): {accuracy:.4f}")
    
    # Macierz Pomyłek (kluczowa dla testera)
    print("\nConfusion Matrix (Rzeczywista vs Przewidziana):\n")
    print(pd.DataFrame(conf_matrix, index=target_names, columns=target_names))
    
    # Szczegółowy Raport Klasyfikacji
    print("\nClassification Report (Precision, Recall, F1 Score - dla każdej klasy):\n")
    print(report)
    
    print("=" * 60)
    print("--- PROJECT FINISHED ---")

# ==============================================================================
# PUNKT WEJŚCIA
# ==============================================================================

if __name__ == "__main__":
    run_cart_project()