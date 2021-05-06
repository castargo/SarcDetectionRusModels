import argparse
import datetime
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, make_scorer,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tabulate import tabulate
from tqdm import tqdm
from xgboost import XGBClassifier

from sarcsdet.configs.sklearn_models_config import *
from sarcsdet.configs.sklearn_models_grid_search_params import *
from sarcsdet.embeddings.gensim_word2vec_ruscorp import \
    GensimWord2VecRUSEmbeddingVectorizer
from sarcsdet.embeddings.NatashaGlove import NatashaGloVeEmbeddingVectorizer
from sarcsdet.embeddings.rubert import RuBERTmbeddingVectorizer
from sarcsdet.models.count_model_metrics import *
from sarcsdet.models.bilstm import BiLSTMClassifier
from sarcsdet.utils import chunks

extra_features = [
    [],
    ['funny_mark'],
    ['interjections'],
    ['exclamation', 'question', 'quotes', 'dotes'],
    ['rating', 'comments_count', 'source',  'submitted_by'],
    [
        'funny_mark', 'interjections', 
        'exclamation', 'question', 'quotes', 'dotes',
        'rating', 'comments_count', 'source',  'submitted_by'
    ],
    ['author', 'subreddit', 'score'],
    [
        'funny_mark', 'interjections', 
        'exclamation', 'question', 'quotes', 'dotes',
        'author', 'subreddit', 'score',
    ]
]
extra_features_strings = ["/".join(f) for f in extra_features]


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--data_path',
                    default='data/Quotes/quotes_ling_feats.pickle',
                    help='path to pickle file with dataset')
parser.add_argument('embedding', choices=['TFIDF', 'NatashaGlove', 'Word2Vec', 'RuBERT'])
parser.add_argument('model', choices=['LogisticRegression', 'XGBoost', 'BernoulliNB', 'Perceptron', 'BiLSTM'])
parser.add_argument('--extra_features', choices=extra_features_strings, default="", help='Choose extra features combinations')
parser.add_argument('--seed', type=int, default=8, help='random seed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for some models with partial fitting')
parser.add_argument('--embedding_path', default=None, help='path to embedding file if required')
parser.add_argument('--data_source', default='quotes', choices=['quotes', 'reddit'])


def tfidf_embedding(train_df, test_df, args):
    """
    return tuple of two TFIDF sparce arrays for train and test dataframes
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3), 
        max_features=50000,
        min_df=2
    )
    if args.data_source == 'quotes':
        X = tfidf.fit_transform(train_df.quote_tokenized)
        X_test = tfidf.transform(test_df.quote_tokenized)
    elif args.data_source == 'reddit':
        X = tfidf.fit_transform(train_df.rus_comment)
        X_test = tfidf.transform(test_df.rus_comment)
    else:
        raise Exception('wrong data_source')

    return X, X_test


def get_features_columns(args):
    if args.data_source == 'quotes':
        if embedding_model.__class__.__name__ == 'RuBERTmbeddingVectorizer':
            columns = ['quote']
        else:
            columns = ['quote_tokenized']
        columns += current_extra_features + ['target']
    elif args.data_source == 'reddit':
        if embedding_model.__class__.__name__ == 'RuBERTmbeddingVectorizer':
            columns = ['rus_comment']
        else:
            columns = ['rus_comment_tokenized']
        
        columns += current_extra_features + ['label']
    else:
        raise Exception('wrong data_source')
    return columns


def fit_chunks_model(train_df, current_extra_features, embedding_model, model, args):
    target_label = 'target' if args.data_source == 'quotes' else 'label'
    classes = train_df[target_label].unique().astype(np.int)
    classes_weights = [
        1 - (train_df[target_label] == 0).sum() / train_df.index.size,
        1 - (train_df[target_label] == 1).sum() / train_df.index.size
    ]
    
    columns = get_features_columns(args)

    if args.model in ['Perceptron']:
        num_epoches = 2
    elif args.model == 'BiLSTM':
        num_epoches = 5
    else:
        num_epoches = 1
    
    data = train_df[columns].values
    pbar = tqdm(total=num_epoches * (train_df.index.size // args.batch_size), desc=f'Fit {args.embedding}_{args.model}')
    for epoch in range(num_epoches):
        for chunk_idx, chunk in enumerate(chunks(data, args.batch_size)):
            text, y = chunk[:, 0], chunk[:, -1]
            y = np.array(y).astype(np.int)
            x = embedding_model.transform(text)

            if len(current_extra_features) > 0:
                extra_features_data = chunk[:, 1:-1].astype(np.float)
                x = np.hstack((x, extra_features_data))

            weights = np.zeros(y.shape)
            weights[y == 0] = classes_weights[0]
            weights[y == 1] = classes_weights[1]

            if args.model in ['LogisticRegression', 'BernoulliNB', 'Perceptron']:
                model.partial_fit(x, y, classes=classes, sample_weight=weights)
            elif args.model == 'BiLSTM':
                model.partial_fit(x, y, sample_weight=weights)
            elif args.model == 'XGBoost':
                if chunk_idx > 0:
                    xgb_model = 'tmp_xgb_model.model'
                else:
                    xgb_model = None
                model.fit(x, y, sample_weight=weights, xgb_model=xgb_model, verbose=10)
                model.save_model('tmp_xgb_model.model')

            pbar.update(1)
            del x
            del y
            del text

        data = shuffle(data, random_state=args.seed)

def predict_chunks_model(test_df, current_extra_features, embedding_model, model, args):
    columns = get_features_columns(args)
    y_pred = []
    y_pred_prob = []
    pbar = tqdm(total=test_df.index.size // args.batch_size, desc=f'Predict {args.embedding}_{args.model}')
    for chunk in chunks(test_df[columns].values, args.batch_size):
        text, y = chunk[:, 0], chunk[:, -1]
        x = embedding_model.transform(text)

        if len(current_extra_features) > 0:
            extra_features_data = chunk[:, 1:-1].astype(np.float)
            x = np.hstack((x, extra_features_data))
        
        pred = model.predict(x)

        if args.model == 'Perceptron':
            pred_prob = pred
        elif args.model == 'BiLSTM':
            pred_prob = pred[:, 0]
            pred = (pred[:, 0] > 0.5).astype(np.int8)
        else:
            pred_prob = model.predict_proba(x)[:, 1]

        y_pred.append(pred)
        y_pred_prob.append(pred_prob)

        pbar.update(1)

    y_pred = np.concatenate(y_pred)
    y_pred_prob = np.concatenate(y_pred_prob)
    return y_pred, y_pred_prob


def fit_model(X, y, model, args):
    model.fit(X, y)


def save_results(y_test, preds, preds_proba, current_extra_features, args):
    results = {}
    results['precision'] = precision_score(y_test, preds)
    results['recall'] = recall_score(y_test, preds)
    results['F1'] = f1_score(y_test, preds)
    results['PR AUC'] = average_precision_score(y_test, preds_proba)
    results['ROC AUC'] = roc_auc_score(y_test, preds_proba)
    
    results_path = Path(f"./fx_results/{args.data_source}")
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f'{args.embedding}_{args.model}_seed-{args.seed}_{datetime.datetime.now()}.json', 'w') as f:
        f.write(json.dumps({
            'embedding': args.embedding,
            'model': args.model,
            'extra_features': current_extra_features,
            'seed': args.seed,
            'results': results,
            'test samples': len(y_test),
        }))

    print(f'{args.embedding} {args.model} seed={args.seed} extra_features={current_extra_features}')
    print(results)
    print(f'Dumped to {str(results_path)}\n')


if __name__ == '__main__':
    args = parser.parse_args()

    current_extra_features = extra_features[extra_features_strings.index(args.extra_features)]

    np.random.seed(args.seed)

    with open(args.data_path, 'rb') as f:
        df = shuffle(pickle.load(f), random_state=args.seed)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=args.seed)

    if args.data_source == 'quotes':
        y_test = test_df.target.values
    elif args.data_source == 'reddit':
        y_test = test_df.label.values
    else:
        raise Exception('wrong data_source')

    if args.embedding in ['TFIDF']:
        if args.data_source == 'quotes':
            y = train_df.target.values
        elif args.data_source == 'reddit':
            y = train_df.label.values
        else:
            raise Exception('wrong data_source')

        if args.embedding == 'TFIDF':
            X, X_test = tfidf_embedding(train_df, test_df, args)
        
        if args.model == 'LogisticRegression':
            model = LogisticRegression(**default_logit_params_rus, random_state=args.seed)
        elif args.model == 'XGBoost':
            model = XGBClassifier(**default_xgb_params_rus, random_state=args.seed)
        elif args.model == 'BernoulliNB':
            model = BernoulliNB(**default_bayes_params_rus)
        elif args.model == 'Perceptron':
            model = Perceptron(n_jobs=-2)
        
        extra_features_data = csr_matrix(train_df[current_extra_features].values.astype(np.float))
        X = hstack([X, extra_features_data])

        extra_test_features_data = csr_matrix(test_df[current_extra_features].values.astype(np.float))
        X_test = hstack([X_test, extra_test_features_data])

        fit_model(X, y, model, args)
        
        preds = model.predict(X_test)
        if args.model == 'Perceptron':
            preds_proba = preds
        else:
            preds_proba = model.predict_proba(X_test)[:, 1]

    elif args.embedding in ['NatashaGlove', 'Word2Vec', 'RuBERT']:
        if args.embedding == 'NatashaGlove':
            if args.embedding_path is None:
                raise Exception('embedding_path have to be lead to NatashaGlove embbeddings')
            embedding_model = NatashaGloVeEmbeddingVectorizer(args.model != 'BiLSTM', path=args.embedding_path)
        elif args.embedding == 'Word2Vec':
            embedding_model = GensimWord2VecRUSEmbeddingVectorizer(args.model != 'BiLSTM')
        elif args.embedding == 'RuBERT':
            if args.embedding_path is None:
                raise Exception('embedding_path have to be lead to RuBERT embbeddings')
            embedding_model = RuBERTmbeddingVectorizer(args.embedding_path)
        
        if args.model == 'LogisticRegression':
            model = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, random_state=args.seed)
        elif args.model == 'XGBoost':
            model = XGBClassifier(**default_xgb_params_rus, random_state=args.seed)
        elif args.model == 'BernoulliNB':
            model = BernoulliNB(**default_bayes_params_rus)
        elif args.model == 'Perceptron':
            model = Perceptron(n_jobs=-2)
        elif args.model == 'BiLSTM':
            model = BiLSTMClassifier((30, 300))
        
        fit_chunks_model(train_df, current_extra_features, embedding_model, model, args)
        preds, preds_proba = predict_chunks_model(test_df, current_extra_features, embedding_model, model, args)

    save_results(y_test, preds, preds_proba, current_extra_features, args)
    if args.model == 'BiLSTM':
        results_path = Path(f"./fx_results/{args.data_source}")
        model.save(str(results_path / f'{args.embedding}_{args.model}_seed-{args.seed}_{datetime.datetime.now()}.h5'))
