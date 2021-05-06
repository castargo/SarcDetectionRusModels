import numpy as np
import pandas as pd

from collections import Counter
from nltk.collocations import *
from nltk.util import ngrams


def get_ngrams(row, n):
    tokens = row.split()
    n_grams = list(ngrams(tokens, n))
    return n_grams


def count_ngrams(df, col, n):
    n_grams = dict()

    for index, row in df.iterrows():
        row_n_grams = get_ngrams(row[col], n)
        for ngram in row_n_grams:
            if ngram in n_grams.keys():
                n_grams[ngram] += 1
            else:
                n_grams[ngram] = 1
                
    return n_grams


def n_common_ngrams(n, ngrams):
    c = Counter(ngrams)
    return c.most_common(n)


def count_metrics(df, column, bigram, N):
    '''
    Полезная статья: 
    http://www.dialog-21.ru/digests/dialog2010/materials/pdf/22.pdf
    '''
    word1, word2 = bigram[0]
    co_freq = bigram[1]
    word1_freq = df[column].str.count(' ' + word1 + ' ').sum()
    word2_freq = df[column].str.count(' ' + word2 + ' ').sum()
    
    mi = np.log2(N * co_freq / (word1_freq * word2_freq))
    tscore = (co_freq - (word1_freq * word2_freq) / N) / np.sqrt(co_freq)
    dice = 2 * co_freq / (word1_freq + word2_freq)
    llh = 2 * co_freq * np.log2(N * co_freq / (word1_freq * word2_freq))
    
    return mi, tscore, dice, llh


def create_metrics_df(df, column, most_common):
    N = len(df.index)
    cols = ['collocation', 'frequency', 'MI', 'T-score', 'Dice', 'LogLikelyHood']
    result_df = pd.DataFrame(columns=cols)
    
    for ppair in most_common:
        mi, tscore, dice, llh = count_metrics(df, column, ppair, N)
        result_df = result_df.append(
            {
                'collocation': ppair[0],
                'frequency': ppair[1],
                'MI': mi,
                'T-score': tscore, 
                'Dice': dice,
                'LogLikelyHood': llh,
            }, ignore_index=True)
        
    return result_df
