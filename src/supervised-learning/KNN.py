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


if __name__ == '__main__':

    df = pd.read_csv('../../dataset/Student_corrected.csv')
    X = df.drop(columns=['GradeClass'])
    y = df['GradeClass']

    # Split dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Contoh Penggunaan KNN Classifier From Scratch\n")

    # 1. Menggunakan jarak Euclidean
    knn_euclidean = KNNClassifier(k=3, metric='euclidean')
    knn_euclidean.fit(X_train, y_train)
    predictions_euclidean = knn_euclidean.predict(X_test)
    print(f"Prediksi dengan Jarak Euclidean (k=3) untuk {X_test}: {predictions_euclidean}")

    # 2. Menggunakan jarak Manhattan
    knn_manhattan = KNNClassifier(k=3, metric='manhattan')
    knn_manhattan.fit(X_train, y_train)
    predictions_manhattan = knn_manhattan.predict(X_test)
    print(f"Prediksi dengan Jarak Manhattan (k=3) untuk {X_test}: {predictions_manhattan}")

    # 3. Menggunakan jarak Minkowski
    knn_minkowski = KNNClassifier(k=3, metric='minkowski', p=4)
    knn_minkowski.fit(X_train, y_train)
    predictions_minkowski = knn_minkowski.predict(X_test)
    print(f"Prediksi dengan Jarak Minkowski (k=3, p=4) untuk {X_test}: {predictions_minkowski}")