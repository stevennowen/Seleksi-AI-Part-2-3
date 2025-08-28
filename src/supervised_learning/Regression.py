import numpy as np

class mySoftmaxRegression:
    def __init__(self, optimizer='gd', learning_rate=0.01, n_iters=1000, regularization=None, lambda_param=0.01):
        if optimizer not in ['gd', 'newton']:
            raise ValueError("Optimizer harus 'gd' atau 'newton'.")
        
        self.optimizer = optimizer
        self.lr = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.class_to_idx_ = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def _add_intercept(self, X):
        # Menambahkan kolom intercept (bias)
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Pemetaan label ke indeks
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_indexed = np.array([self.class_to_idx_[label] for label in y])
        y_one_hot = self._one_hot(y_indexed, n_classes)

        if self.optimizer == 'newton':
            X_aug = self._add_intercept(X)
            self.weights = np.zeros((n_features + 1, n_classes))
            self._newtons_method(X_aug, y_one_hot)
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)
            self._gradient_descent(X, y_one_hot)

    def _gradient_descent(self, X, y_one_hot):
        n_samples = X.shape[0]
        n_classes = y_one_hot.shape[1]
        
        for i in range(self.n_iters):
            # Langkah maju (Forward pass)
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(linear_model)
            
            # Hitung gradien
            error = y_pred - y_one_hot
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0)
            
            # Tambahkan regularisasi
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            
            # Perbarui parameter
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % 100 == 0:
                loss = self._compute_loss(y_pred, y_one_hot, X)
                print(f"Iterasi {i}: Loss = {loss:.6f}")

    def _newtons_method(self, X, y_one_hot):
        n_samples, n_features = X.shape
        n_classes = y_one_hot.shape[1]
        
        for i in range(self.n_iters):
            # Langkah maju (Forward pass)
            linear_model = np.dot(X, self.weights)
            probas = self._softmax(linear_model)
            
            # Hitung gradien
            gradient = (1 / n_samples) * np.dot(X.T, (probas - y_one_hot))
            
            # Tambahkan regularisasi
            if self.regularization == 'l2':
                gradient += (self.lambda_param / n_samples) * self.weights
            
            # Komputasi Hessian dengan aproksimasi per kelas
            for c in range(n_classes):
                p_c = probas[:, c]
                diag_w = p_c * (1 - p_c)
                X_weighted = X * diag_w[:, np.newaxis]
                hessian_c = np.dot(X_weighted.T, X) / n_samples
                
                if self.regularization == 'l2':
                    hessian_c += (self.lambda_param / n_samples) * np.eye(n_features)
                
                try:
                    # Invers Hessian untuk kelas ini
                    hessian_inv_c = np.linalg.inv(hessian_c)
                    # Perbarui bobot untuk kelas ini
                    self.weights[:, c] -= self.lr * np.dot(hessian_inv_c, gradient[:, c])
                except np.linalg.LinAlgError:
                    # Kembali ke gradient descent jika Hessian singular
                    self.weights[:, c] -= self.lr * gradient[:, c]
            
            if i % 100 == 0:
                loss = self._compute_loss(probas, y_one_hot, X)
                print(f"Iterasi {i}: Loss = {loss:.6f}")

    def _compute_loss(self, probas, y_one_hot, X):
        probas_clipped = np.clip(probas, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_one_hot * np.log(probas_clipped), axis=1))
        
        if self.regularization == 'l2':
            if self.optimizer == 'newton':
                # Untuk Newton, intercept sudah termasuk di dalam bobot
                reg_term = (self.lambda_param / (2 * X.shape[0])) * np.sum(self.weights**2)
            else:
                # Untuk GD, hanya bobot yang diregularisasi
                reg_term = (self.lambda_param / (2 * X.shape[0])) * np.sum(self.weights**2)
            loss += reg_term
            
        return loss

    def predict(self, X):
        if self.optimizer == 'newton':
            X = self._add_intercept(X)
            linear_model = np.dot(X, self.weights)
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            
        probas = self._softmax(linear_model)
        predicted_indices = np.argmax(probas, axis=1)
        predicted_labels = self.classes_[predicted_indices]
        return predicted_labels