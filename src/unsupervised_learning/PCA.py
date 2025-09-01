import numpy as np

class myPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. Standarisasi 
        self.mean = np.mean(X, axis=0)
        X_std = X - self.mean
        
        # 2. covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # 3. Hitung eigenvectors dan eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Urutkan eigenvectors berdasarkan eigenvalues 
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 5. Pilih semua eigenvectors
        self.components = eigenvectors

        # 6. explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues[:X.shape[1]]
        self.explained_variance_ratio_ = explained_variance / total_variance


    def fit_transform(self, X):

        # 1. Standarisasi
        self.mean = np.mean(X, axis=0)
        X_std = X - self.mean
        
        # 2. covariance matrix
        cov_matrix = np.cov(X_std, rowvar=False)
        
        # 3. eigenvectors dan eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Urutkan eigenvectors berdasarkan eigenvalues
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 5. k eigenvectors pertama
        self.components = eigenvectors[0:self.n_components]
        
        # 6. explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = explained_variance / total_variance
        
        # 7. Transformasikan data asli ke dimensi yang lebih rendah
        X_transformed = np.dot(X_std, self.components.T)
        
        return X_transformed
