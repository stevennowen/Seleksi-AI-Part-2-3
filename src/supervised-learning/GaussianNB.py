import numpy as np

class GaussianNaiveBayes:

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # mean, varians, dan prior untuk setiap kelas
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            # frekuensi kemunculan setiap kelas
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):

        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # epsilon kecil jika varians adalah 0
        epsilon = 1e-9
        numerator = np.exp(- (x - mean)**2 / (2 * var + epsilon))
        denominator = np.sqrt(2 * np.pi * var + epsilon)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []

        # posterior probability untuk setiap kelas
        for idx, c in enumerate(self._classes):
            # Log dari prior
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._pdf(idx, x)))
            # Digabung untuk dapat posterior
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Dipilih kelas dengan posterior probability tertinggi
        return self._classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification

    # Membuat dataset sintetis untuk klasifikasi
    X, y = make_classification(n_samples=150, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, flip_y=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print("Contoh Penggunaan Gaussian Naive Bayes Classifier From Scratch\n")

    # Inisialisasi dan latih model
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # Lakukan prediksi
    predictions = gnb.predict(X_test)

    print(f"Data Uji:\n{X_test[:5]}")
    print(f"Prediksi:\n{predictions[:5]}")
    print(f"Label Asli:\n{y_test[:5]}")
    
    # Evaluasi akurasi
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAkurasi model: {accuracy * 100:.2f}%")