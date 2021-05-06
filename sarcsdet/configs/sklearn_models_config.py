default_logit_params_rus = { 
    'class_weight': 'balanced', 'multi_class': 'ovr', 'n_jobs': -2
}

logit_params_rus = {
    'C': 0.1, 'solver': 'newton-cg', 'penalty': 'l2', 
    'class_weight': 'balanced', 'multi_class': 'ovr', 'max_iter': 5000, 'n_jobs': -2,
}

default_bayes_params_rus = {
}

bayes_params_rus = {
    'alpha': 0.7, 'fit_prior': True, 'class_prior': [0.5, 0.5]
}

default_xgb_params_rus = {
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'eval_metric': ['map', 'aucpr'],
    'n_jobs': -2,
    'n_estimators': 1000,
    'max_depth': 5,
    'num_parallel_tree': 10,
    'base_score': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
}

xgb_params_rus = {
    'n_estimators': 1000, 'scale_pos_weight': 13.5,  # (y == 0).sum() / (y == 1).sum()
    'min_child_weight': 1, 'gamma': 5, 'max_depth': 3, 'use_label_encoder': False,
    'objective': 'binary:logistic',
    'num_parallel_tree': 5, 'learning_rate': 0.3, 'eval_metric': ['map', 'aucpr'],
    'base_score': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'n_jobs': -2,
}

svm_params_rus = {
    'C': 1, 'gamma': 'scale', 'degree': 9, 'kernel': 'rbf'
}
