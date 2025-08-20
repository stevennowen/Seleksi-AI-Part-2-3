import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class KNNClassifier:

    def __init__(self, k=5, metric='euclidean', p=3):

        if k <= 0:
            raise ValueError("K (jumlah tetangga) harus lebih besar dari 0.")
        if metric not in ['euclidean', 'manhattan', 'minkowski']:
            raise ValueError("Metric harus 'euclidean', 'manhattan', atau 'minkowski'.")
        
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _calculate_distance(self, p1, p2):
       
        if self.metric == 'euclidean':
            # euclidean
            return np.sqrt(np.sum((p1 - p2)**2))
        
        elif self.metric == 'manhattan':
            # manhattan
            return np.sum(np.abs(p1 - p2))
        
        elif self.metric == 'minkowski':
            # minkowski
            return np.power(np.sum(np.abs(p1 - p2)**self.p), 1/self.p)

    
    def _predict_single(self, x_test):

        # 1. Hitung jarak dari titik uji ke semua titik training
        distances = [self._calculate_distance(x_test, x_train) for x_train in self.X_train]
        
        # 2. Dapatkan indeks dari k tetangga terdekat
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Mendapatkan label dari k tetangga terdekat
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Tentukan label yang paling sering muncul
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)