import numpy as np

class myDBSCAN:

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', p=3):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels_ = None 

    def _get_distance(self, p1, p2):
        if self.metric == 'euclidean':
            return np.linalg.norm(p1 - p2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(p1 - p2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(p1 - p2)**self.p), 1/self.p)
        else:
            raise ValueError("Metric tidak valid. Pilih 'euclidean', 'manhattan', atau 'minkowski'.")

    def fit(self, X):

        n_samples = X.shape[0]
        # Inisialisasi semua label sebagai unvisited (0)
        self.labels_ = np.full(n_samples, 0)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != 0:
                continue
            
            # tetangga dari titik saat ini
            neighbors_indices = self._get_neighbors(X, i)
            
            # Jika jumlah tetangga kurang dari min_samples, tandai sebagai noise (sementara)
            if len(neighbors_indices) < self.min_samples:
                self.labels_[i] = -1 
            else:
                # Buat cluster baru
                cluster_id += 1
                self._expand_cluster(X, i, neighbors_indices, cluster_id)

    def _get_neighbors(self, X, point_index):
        neighbors = []
        for i in range(X.shape[0]):
            if self._get_distance(X[point_index], X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, point_index, neighbors_indices, cluster_id):

        self.labels_[point_index] = cluster_id
        
        i = 0
        while i < len(neighbors_indices):
            current_neighbor_index = neighbors_indices[i]
            
            # Jika tetangga adalah noise, jadikan border point dari cluster saat ini
            if self.labels_[current_neighbor_index] == -1:
                self.labels_[current_neighbor_index] = cluster_id

            # Proses jika tetangga ini belum dikunjungi
            elif self.labels_[current_neighbor_index] == 0:
                self.labels_[current_neighbor_index] = cluster_id
                
                # tetangga dari tetangga ini
                new_neighbors = self._get_neighbors(X, current_neighbor_index)
                
                # Jika tetangga ini juga core point, tambahkan tetangganya ke antrian
                if len(new_neighbors) >= self.min_samples:
                    neighbors_indices.extend(new_neighbors)
            
            i += 1
