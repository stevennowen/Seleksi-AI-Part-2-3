import numpy as np

class myKMeans:

    def __init__(self, n_clusters=3, max_iters=100, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method
        self.centroids = None

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        
        if self.init_method == 'random':
            # Pilih k sampel acak dari data sebagai centroid awal
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_indices]
            
        elif self.init_method == 'kmeans++': # BONUS
            # 1. Pilih centroid pertama secara acak
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                # 2. Hitung kuadrat jarak dari setiap titik ke centroid terdekat
                dist_sq = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in X])
                
                # 3. Pilih titik data baru sebagai centroid dengan probabilitas sebanding dengan kuadrat jaraknya
                probs = dist_sq / dist_sq.sum()
                cumulative_probs = probs.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[j])
                        break
            self.centroids = np.array(centroids)

    def fit(self, X):
        self._initialize_centroids(X)

        for _ in range(self.max_iters):
            # 1. Tentukan cluster untuk setiap titik data
            labels = self._assign_clusters(X)

            # 2. Perbarui posisi centroid
            new_centroids = self._update_centroids(X, labels)

            # (Cek konvergensi) jika centroid tidak berubah, hentikan iterasi
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
        
        self.labels_ = self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
        return new_centroids

    def predict(self, X):
        return self._assign_clusters(X)
