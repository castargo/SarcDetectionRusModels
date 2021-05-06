import numpy as np
from deeppavlov import build_model, configs
from deeppavlov.core.common.file import read_json
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import pickle
import numpy as np


class RuBERTmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 path='./data/Embeddings/rubert_cased_L-12_H-768_A-12_pt/',
                 data_path='./data/Embeddings/rubert_batchs.pickle'):
        bert_config = read_json(configs.embedder.bert_embedder)
        bert_config['metadata']['variables']['BERT_PATH'] = path
        
        self.seq_size = 30
        
        self.model = build_model(bert_config, download=False)

        self.data_path = data_path
        path = Path(self.data_path)
        if path.is_file():
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = {}

    def fit(self, X, y=None):
        return self

    def get_params(self, **params):
        return {'model': self.model}
    
    def _transform(self, X):
        return self.model(X)
    
    def full_embedding_transform(self, X):
        e1, e2, _, _, _, _, _ = self.model(X)
        results = np.zeros((len(e2), self.seq_size, e2[0].shape[-1]))
        
        for i, e in enumerate(e2):
            results[i][:len(e)] = e[:self.seq_size]
            
        return results
    
    def transform(self, X):
        X = tuple(X)
        if X in self.data:
            return self.data[X]
        else:
            _, _, _, _, _, _, e = self._transform(X)
            self.data[X] = e

            with open(self.data_path, 'wb') as f:
                pickle.dump(self.data, f)

            return e
