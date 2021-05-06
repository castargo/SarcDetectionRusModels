import gensim

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X = [val.split() for val in X.to_list()]
        self.model = gensim.models.Word2Vec(X, size=100)
        self.word2vec = dict(zip(self.model.wv.index2word, self.model.wv.vectors))
        self.word2weight = None
        
        if len(self.word2vec):
            self.dim = next(iter(self.word2vec.values())).shape[0]
        else:
            self.dim = 0

        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self
    
    def get_params(self, **params):
        return {}
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
