import pickle
from pathlib import Path

import gensim
import numpy as np
from joblib import Parallel, delayed
from navec import Navec
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class NatashaGloVeEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self, concat_strings=True, path='../data/Embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.path = path
        self.concat_strings = concat_strings
        
        self.navec = Navec.load(self.path)
        self.seq_size = 30


    def fit(self, X, y=None):
        return self

    def get_params(self, **params):
        return {'navec': self.navec, 'seq_size': self.seq_size}


    def _transform_one(self, input_string):
        tokens = input_string.split()
        embedd_tokens = [[self.navec.get(t, self.navec['<unk>'])] for t in tokens[:self.seq_size]]
        if len(embedd_tokens) < self.seq_size:
            embedd_tokens += [[self.navec['<pad>']]] * (self.seq_size - len(embedd_tokens))
        
        if self.concat_strings:
            return np.concatenate(embedd_tokens, 1)
        else:
            return np.array([embedd_tokens])

    def transform(self, X):
        vectors = [self._transform_one(s) for s in X]
        embeddings = np.concatenate(vectors, 0)
        if self.concat_strings:
            return embeddings
        else:
            return np.squeeze(embeddings, 2)