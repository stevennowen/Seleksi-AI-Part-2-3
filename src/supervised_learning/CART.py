import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Nilai prediksi jika ini adalah leaf node

class myDecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Cek kondisi berhenti
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features_, replace=False)

        # Cari split terbaik
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Jika tidak ada split yang menguntungkan, buat leaf node
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Buat split dan bangun sub-tree secara rekursif
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):

        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._gini_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        
        return split_idx, split_thresh

    def _gini_gain(self, y, X_column, split_thresh):

        # Gini Impurity parent
        parent_gini = self._gini_impurity(y)

        # Buat split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted average Gini Impurity dari children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r

        gg = parent_gini - child_gini
        return gg

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini_impurity(self, y):
        # Menghitung proporsi setiap kelas
        hist = np.bincount(y)
        ps = hist / len(y)
        # Rumus Gini: 1 - sum(p^2)
        return 1 - np.sum([p**2 for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        if not counter:
            return None
        value = counter.most_common(1)[0][0]
        return value

    #Prediksi Tunggal
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def get_params(self, deep=True):
        return {
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self