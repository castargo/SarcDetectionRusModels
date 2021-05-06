import gensim

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class MeanW2VEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        X = [val.split() for val in X.to_list()]
        self.model = gensim.models.Word2Vec(X, size=100)
        self.word2vec = dict(zip(self.model.wv.index2word, self.model.wv.vectors))

        if self.word2vec:
            self.dim = next(iter(self.word2vec.values())).shape[0]
        else:
            self.dim = 0
        return self
    
    def get_params(self, **params):
        return {}

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
