import numpy as np

class myPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X):

        # 1. Standarisasi data
        self.mean = np.mean(X, axis=0)
        X_std = X - self.mean
        
        # 2. Hitung covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # 3. Hitung eigenvectors dan eigenvalues dari covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Urutkan eigenvectors berdasarkan eigenvalues (terbesar ke terkecil)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 5. Pilih k eigenvectors pertama (k = n_components)
        self.components = eigenvectors[0:self.n_components]
        
        # 6. Hitung explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = explained_variance / total_variance
        
        # 7. Transformasikan data asli ke ruang dimensi yang lebih rendah
        X_transformed = np.dot(X_std, self.components.T)
        
        return X_transformed
