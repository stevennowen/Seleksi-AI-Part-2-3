import numpy as np

class myGaussianNaiveBayes:

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
    
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
