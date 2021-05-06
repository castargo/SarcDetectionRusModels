import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


with open("./Embeddings/crawl-300d-2M.vec", "rb") as lines:
    ft = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

    
class MeanFastTextEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.ft = ft
        self.dim = 300

    def fit(self, X, y=None):
        return self

    def get_params(self, **params):
        return {}

    def transform(self, X):
        return np.array([
            np.mean([self.ft[w] for w in words if w in self.ft] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    