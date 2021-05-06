import gensim.downloader as api
import numpy as np
from joblib import Parallel, delayed
from pymystem3 import Mystem
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import pickle
from tqdm import tqdm


def tag(word):
    RNC2UPOS = {
        'A': 'ADJ',
        'ADV': 'ADV',
        'ADVPRO': 'ADV',
        'ANUM': 'ADJ',
        'APRO': 'DET',
        'COM': 'ADJ',
        'CONJ': 'SCONJ',
        'INTJ': 'INTJ',
        'NONLEX': 'X',
        'NUM': 'NUM',
        'PART': 'PART',
        'PR': 'ADP',
        'S': 'NOUN',
        'SPRO': 'PRON',
        'UNKN': 'X',
        'V': 'VERB'
    }
    m = Mystem()
    processed = m.analyze(word)[0]
    lemma = processed["analysis"][0]["lex"].lower().strip()
    pos = processed["analysis"][0]["gr"].split(',')[0]
    pos = pos.split('=')[0].strip()
    tagged = lemma+'_'+RNC2UPOS[pos]
    return tagged


class GensimWord2VecRUSEmbeddingVectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self, concat_strings=True, tag_path='./data/Embeddings/word2vec_ruscorpora_tags.pickle'):
        self.model = api.load("word2vec-ruscorpora-300")
        self.seq_size = 30
        self.concat_strings = concat_strings
        
        self.vec_size = 300

        self.tag_path = tag_path
        self._load_tag_tokens()

    def fit(self, X, y=None):
        return self

    def get_params(self, **params):
        return {'model': self.model, 'seq_size': self.seq_size}

    def _load_tag_tokens(self):
        if Path(self.tag_path).is_file():
            with open(self.tag_path, 'rb') as f:
                self.tag_tokens = pickle.load(f)
        else:
            self.tag_tokens = {}

    def _get_token(self, origin_token):
        if origin_token in self.tag_tokens:
            return self.tag_tokens[origin_token]
        
        try:
            token = tag(origin_token)
            self.tag_tokens[origin_token] = token
        except:
            self.tag_tokens[origin_token] = "<unk>"

        return self.tag_tokens[origin_token]
    
    def _get_token_vec(self, origin_token):
        token = self._get_token(origin_token)
        if token in self.model:
            return self.model[token]
        else:
            return np.zeros((self.vec_size, ))
    
    def _transform_one(self, input_string):
        tokens = input_string.split()
        embedd_tokens = [
            np.expand_dims(self._get_token_vec(t), 0)
            for t in tokens[:self.seq_size]
        ]
        if len(embedd_tokens) < self.seq_size:
            embedd_tokens += [np.expand_dims(np.zeros((self.vec_size, )), 0)] * (self.seq_size - len(embedd_tokens))
        
        if self.concat_strings:
            return np.concatenate(embedd_tokens, 1)
        else:
            return np.array([embedd_tokens])

    def transform(self, X):
        vectors = [self._transform_one(s) for s in X]

        with open(self.tag_path, 'wb') as f:
            pickle.dump(self.tag_tokens, f)
        
        embeddings = np.concatenate(vectors, 0)
        if self.concat_strings:
            return embeddings
        else:
            return np.squeeze(embeddings, 2)
