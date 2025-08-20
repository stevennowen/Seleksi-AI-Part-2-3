import numpy as np

class mySoftmaxRegression:

    def __init__(self, optimizer='gd', learning_rate=0.01, n_iters=1000, regularization=None, lambda_param=0.1):
        if optimizer not in ['gd', 'newton']:
            raise ValueError("Optimizer harus 'gd' atau 'newton'.")
        
        self.optimizer = optimizer
        self.lr = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None # Untuk Newton's method, bias digabungkan ke weights

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.optimizer == 'newton':
            X = self._add_intercept(X)
            self.weights = np.zeros((n_features + 1, n_classes))
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)

        y_one_hot = self._one_hot(y, n_classes)

        # Pilih metode optimasi
        if self.optimizer == 'gd':
            self._gradient_descent(X, y_one_hot)
        elif self.optimizer == 'newton':
            self._newtons_method(X, y_one_hot)

    def _gradient_descent(self, X, y_one_hot):
        n_samples = X.shape[0]
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._softmax(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_one_hot))
            db = (1 / n_samples) * np.sum(y_predicted - y_one_hot, axis=0)
            
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def _newtons_method(self, X, y_one_hot):
        n_samples = X.shape[0]
        n_features, n_classes = self.weights.shape

        for i in range(self.n_iters):
            # 1. Hitung probabilitas
            linear_model = np.dot(X, self.weights)
            probas = self._softmax(linear_model)

            # 2. Hitung gradien (turunan pertama)
            gradient = (-1 / n_samples) * np.dot(X.T, (y_one_hot - probas))
            if self.regularization == 'l2':
                gradient += (self.lambda_param / n_samples) * self.weights
            
            # 3. Hitung matriks Hessian (turunan kedua)
            hessian = np.zeros((n_features, n_features, n_classes))
            for c in range(n_classes):
                # Diagonal matrix R_c
                r_c = np.diag(probas[:, c] * (1 - probas[:, c]))
                # Hessian untuk kelas c
                hessian[:, :, c] = (1 / n_samples) * X.T @ r_c @ X
                if self.regularization == 'l2':
                    hessian[:, :, c] += (self.lambda_param / n_samples) * np.eye(n_features)
            
            # 4. Update bobot untuk setiap kelas secara terpisah
            # W_new = W_old - H^-1 * g
            for c in range(n_classes):
                try:
                    # Invers Hessian
                    h_inv = np.linalg.inv(hessian[:, :, c])
                    # Update bobot untuk kelas c
                    self.weights[:, c] -= h_inv @ gradient[:, c]
                except np.linalg.LinAlgError:
                    # Fallback ke Gradient Descent jika Hessian singular (tidak bisa di-invers)
                    print(f"Iterasi {i}, kelas {c}: Hessian singular, fallback ke GD.")
                    self.weights[:, c] -= self.lr * gradient[:, c]
            
            # Print Process
            if i % 10 == 0:
                loss = -np.mean(np.sum(y_one_hot * np.log(probas + 1e-9), axis=1))
                print(f"Iterasi {i}: Loss = {loss:.4f}")

    def predict(self, X):
        if self.optimizer == 'newton':
            X = self._add_intercept(X)
            linear_model = np.dot(X, self.weights)
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            
        probas = self._softmax(linear_model)
        return np.argmax(probas, axis=1)

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=1,
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print("--- 1. Optimasi dengan Gradient Descent ---")
    model_gd = mySoftmaxRegression(optimizer='gd', learning_rate=0.1, n_iters=500, regularization='l2')
    model_gd.fit(X_train, y_train)
    predictions_gd = model_gd.predict(X_test)
    accuracy_gd = accuracy_score(y_test, predictions_gd)
    print(f"Akurasi dengan Gradient Descent: {accuracy_gd * 100:.2f}%\n")

    print("--- 2. Optimasi dengan Newton's Method (Bonus) ---")
    model_newton = mySoftmaxRegression(optimizer='newton', n_iters=50, regularization='l2', learning_rate=0.1)
    model_newton.fit(X_train, y_train)
    predictions_newton = model_newton.predict(X_test)
    accuracy_newton = accuracy_score(y_test, predictions_newton)
    print(f"\nAkurasi dengan Newton's Method: {accuracy_newton * 100:.2f}%")