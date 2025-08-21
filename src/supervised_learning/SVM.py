import numpy as np

class mySVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel if kernel is not None else self.linear_kernel
        self.weights = None
        self.bias = None

    # Fungsi Kernel
    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)
        
    @staticmethod
    def polynomial_kernel(x1, x2, p=3, c=1):
        return (np.dot(x1, x2.T) + c) ** p

    @staticmethod
    def rbf_kernel(x1, x2, gamma=0.1):
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        
        # jarak Euclid
        dist_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-gamma * dist_sq)

    def fit(self, X, y):

        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_samples)
        self.bias = 0

        K = self.kernel(X, X)

        # training dengan gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Kondisi untuk update bobot (berdasarkan hinge loss)
                # f(x_i) = w.x_i - b 
                condition = y_[idx] * (np.dot(self.weights, K[:, idx]) - self.bias) >= 1
                
                if condition:
                    # Update bobot hanya dengan regularisasi jika klasifikasi benar dan di luar margin
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    # Update bobot dengan regularisasi dan loss jika klasifikasi salah atau di dalam margin
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - y_[idx] * K[:, idx])
                    self.bias -= self.lr * y_[idx]
        self.X_train = X

    def predict(self, X):

        K = self.kernel(self.X_train, X)
        
        # hitung output
        linear_output = np.dot(self.weights, K) - self.bias
        
        return np.sign(linear_output).astype(int)