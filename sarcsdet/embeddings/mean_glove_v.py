import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


with open("../data/Embeddings/glove.6B.300d.txt", "rb") as lines:
    glove = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    

class MeanGloVeEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.glove = glove
        self.dim = 300

    def fit(self, X, y=None):
        return self

    def get_params(self, **params):
        return {}

    def transform(self, X):
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])    
