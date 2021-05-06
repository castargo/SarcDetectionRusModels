import numpy as np


# Logistic Regression
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
lr_grid = [
    {'C': C, 'penalty': ['l1'], 'solver': ['liblinear']},
    {'C': C, 'penalty': ['l2', 'none'], 'solver': ['lbfgs', 'newton-cg', 'sag']},
    # {'C': C, 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0, 0.2, 0.5, 0.7, 1]}
]
# clf = LogisticRegression(max_iter=1000, multi_class='ovr', class_weight='balanced')

# Bernoulli Naive Bayes
nb_grid = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
    'fit_prior': [True, False], 
    'class_prior': [None, [.1,.9], [.2,.8], [.3,.7], [.4,.6], [.5,.5]]
}
# clf = BernoulliNB()

# XGBoost
xgb_grid = {
    'objective': ['binary:logistic'],
    'min_child_weight': [1, 3, 5],
    'max_depth': [2, 3, 4],
    'num_parallel_tree': [3, 5, 7],
    'reg_alpha': [0.5],
    'reg_lambda': [0.5]
}
# clf = XGBClassifier(
#     n_estimators=1000, use_label_encoder=False, eval_metric=['map', 'aucpr'],
#     scale_pos_weight=(y == 0).sum() / (y == 1).sum()
# )

# SVM
svm_grid = {
    'C': [100, 500, 1000],
    'gamma': ['scale', 'auto'],
    'degree': [3, 5, 9, 12],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'decision_function_shape': ['ovo', 'ovr']
}
# clf = SVC(
#     class_weight={0: df['target'].value_counts(normalize=True)[0], 1: df['target'].value_counts(normalize=True)[1]}, 
#     probability=False
# )
