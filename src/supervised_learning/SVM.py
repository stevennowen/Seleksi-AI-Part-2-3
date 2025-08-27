import numpy as np

class mySVM:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000, kernel='linear', gamma='scale'):
        self.lr = learning_rate
        self.C = C 
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.gamma = gamma
        
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.classifiers_ = []  # Untuk klasifikasi multiclass
        self.class_weights_ = []  # Untuk multiclass
        self.class_biases_ = []  # Untuk multiclass
        
        # Set fungsi kernel
        if kernel == 'linear':
            self.kernel = self.linear_kernel
        elif kernel == 'rbf':
            self.kernel = self.rbf_kernel
        else:
            raise ValueError("Kernel harus 'linear' atau 'rbf'")

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
        
    def rbf_kernel(self, x1, x2):
        if self.gamma == 'scale':
            gamma = 1.0 / (x1.shape[1] * x1.var()) if x1.shape[1] > 0 and x1.var() > 0 else 0.1
        elif self.gamma == 'auto':
            gamma = 1.0 / x1.shape[1] if x1.shape[1] > 0 else 0.1
        else:
            gamma = self.gamma
            
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
            
        # Hitung jarak Euclidean kuadrat
        dist_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-gamma * dist_sq)

    def fit(self, X, y):
        # Simpan data training
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        
        n_samples, n_features = X.shape
        
        if len(self.classes_) == 2:
            # Klasifikasi biner: ubah label menjadi -1 dan 1
            y_binary = np.where(y == self.classes_[0], -1, 1)
            self._fit_binary(X, y_binary)
        else:
            # Klasifikasi multiclass menggunakan one-vs-rest
            self._fit_multiclass(X, y)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Precompute matrix kernel
        K = self.kernel(X, X)
        
        # Gradient descent untuk SVM
        for epoch in range(self.n_iters):
            for i in range(n_samples):
                # Hitung prediksi
                prediction = np.sum(self.alpha * y * K[i, :]) + self.b
                
                if y[i] * prediction < 1:
                    # Update alpha dan b
                    self.alpha[i] += self.lr * (1 - y[i] * prediction - 2 * (1/self.C) * self.alpha[i])
                    self.b += self.lr * y[i]
                else:
                    # Hanya update term regularisasi
                    self.alpha[i] -= self.lr * 2 * (1/self.C) * self.alpha[i]
            
            self.alpha = np.clip(self.alpha, 0, self.C)
            
        
        # Simpan support vectors (vektor dengan alpha > 0)
        support_vector_indices = self.alpha > 1e-5
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y[support_vector_indices]
        self.support_vector_alphas_ = self.alpha[support_vector_indices]

    def _fit_multiclass(self, X, y):
        n_classes = len(self.classes_)
        
        self.classifiers_ = []
        
        for i, class_label in enumerate(self.classes_):
            print(f"Training classifier untuk kelas {class_label} ({i+1}/{n_classes})")
            
            y_binary = np.where(y == class_label, 1, -1)
            
            # Train classifier biner
            svm_binary = mySVM(
                learning_rate=self.lr,
                C=self.C,
                n_iters=self.n_iters,
                kernel=self.kernel_type,
                gamma=self.gamma
            )
            
            # Set kelas untuk classifier biner
            svm_binary.classes_ = np.array([-1, 1])
            svm_binary.fit(X, y_binary)
            
            self.classifiers_.append(svm_binary)

    def _calculate_accuracy(self, X, y):
        try:
            predictions = self.predict(X)
            return np.mean(predictions == y)
        except:
            return 0.0  # Return 0 jika prediksi gagal

    def decision_function(self, X):
        if not hasattr(self, 'classes_') or self.classes_ is None:
            raise ValueError("Model belum di-training. Panggil fit() terlebih dahulu.")
            
        if len(self.classes_) == 2:
            # Klasifikasi biner
            if self.alpha is None or self.X_train is None:
                return np.zeros(X.shape[0])
                
            K = self.kernel(self.X_train, X)
            return np.dot(self.alpha * self.y_train, K) + self.b
        else:
            # Klasifikasi multiclass
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            scores = np.zeros((n_samples, n_classes))
            
            for i, clf in enumerate(self.classifiers_):
                scores[:, i] = clf.decision_function(X)
            
            return scores

    def predict(self, X):
        # Validasi model sudah trained
        if self.classes_ is None:
            raise ValueError("Model belum di-training. Panggil fit() terlebih dahulu.")
            
        if len(self.classes_) == 2:
            # Klasifikasi biner
            scores = self.decision_function(X)
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            # Klasifikasi multiclass
            scores = self.decision_function(X)
            return self.classes_[np.argmax(scores, axis=1)]

    def get_params(self, deep=True):
        return {
            'learning_rate': self.lr,
            'C': self.C,
            'n_iters': self.n_iters,
            'kernel': self.kernel_type,
            'gamma': self.gamma
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)