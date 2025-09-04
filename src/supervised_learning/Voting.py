import numpy as np
from collections import Counter

class mySoftVotingClassifier:
    def __init__(self, estimators, voting='soft', weights=None):

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.classes_ = None
        
    def fit(self, X, y):
        for name, estimator in self.estimators:
            estimator.fit(X, y)
        
        # Mengambil classes dari estimator pertama
        self.classes_ = self.estimators[0][1].classes_
        return self
    
    def predict_proba(self, X):
        probas = []
        
        for name, estimator in self.estimators:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                if estimator.optimizer == 'newton':
                    X_aug = estimator._add_intercept(X)
                    linear_model = np.dot(X_aug, estimator.weights)
                else:
                    linear_model = np.dot(X, estimator.weights) + estimator.bias
                proba = estimator._softmax(linear_model)
            probas.append(proba)
        
        # Rata-rata probabilitas
        if self.weights is not None:
            weighted_probas = [w * p for w, p in zip(self.weights, probas)]
            avg_proba = np.sum(weighted_probas, axis=0) / np.sum(self.weights)
        else:
            avg_proba = np.mean(probas, axis=0)
            
        return avg_proba
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def get_params(self, deep=True):
        return {
            'estimators': self.estimators,
            'voting': self.voting,
            'weights': self.weights
        }