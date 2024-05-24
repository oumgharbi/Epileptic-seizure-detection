# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:52:05 2023

Run this script to perform a nested cross-validation for epileptic seizure detection.
In the inner loop of nested CV, a grid-search is performed to find optimal hyperparameters.
The best hyperparameters from each inner fold are then used to test the algorithm on a held-out test set of patients.
The estimator and hyperparameter grid are free parameters to be selected by the user.
Performance is measured and optimized according to f1-score.
Number of inner and outer folds (k, k_inner) impact the number of patients to be used for training,
validation and test.

Size of the hyperparameter grid is restricted by computational cost.

@author: Oumayma Gharbi
"""

import itertools
from utils.modules import *


# old_stdout = sys.stdout  #replaced with logger

# model = SVC()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = xgb.XGBClassifier()

# SVM ----------------------------------------
kernel = ['linear', 'rbf']
c = [0.001, 0.01, 0.1, 1]
# --------------------------------------------

# Logistic Regression ------------------------
solver = ['newton-cg', 'liblinear']
# c = [0.001, 0.01, 0.1, 1 ]
# --------------------------------------------

# Decision Tree ------------------------------
splitter = ['best']  # Decision Tree
criterion = ['gini', 'entropy', 'log_loss']
# --------------------------------------------

# XGBoost ------------------------------------
# max_depth = [3, 5, 7, 9, 11]
# max_depth = [5, 7, 9, 11]
# min_child_weight = [1, 3, 5, 7]
max_depth = [7, 9, 11]
min_child_weight = [5, 7]
# --------------------------------------------


def f(x):
    return {'XGBClassifier': (max_depth, min_child_weight),
            'DecisionTreeClassifier': (splitter, criterion),
            'LogisticRegression': (solver, c),
            'SVC': (kernel, c)
            }[x]


(hyperparam1, hyperparam2) = f(model.__class__.__name__)

# Select one window size and step at a time
# win_size_range = [10, 15, 20, 25]
win_size = 25
# step_range = [3, 5]
step = 5

# To identify the range of tau and threshold to be optimized
# the longest combination (tau, thresh) should be shorter than the shortest seizure
# e.g.,  (tau=8 x threshold=0.85) = 6.8,
# so, at least 7 of 8 observations should be classified as positive
# in order to create and event (after regularization)
# for a step of 5 sec, 7 observations are 30 sec long
# which is shorter than the shortest FBTCS, (duration=52 sec)

# Adjust regularization parameters according to your data

# tau_range = [5, 6, 7, 8, 9, 10]
tau_range = [4, 6, 8]

# thresh_range = [0.75, 0.8, 0.85, 0.9, 0.95]
thresh_range = [0.65, 0.75, 0.85]

combin = [hyperparam1, hyperparam2, [win_size], [step], tau_range, thresh_range]
all_combinations = list(itertools.product(*combin))

t1 = datetime.datetime.now()
tmp = datetime.datetime.strftime(t1, '%Y%m%d_%H%M%S')

# out_path = f'Oumayma/Hexoskin/NestedCV/FBTCS_patients/{model.__class__.__name__}/win{win_size}_step{step}'
# log_file = open(home + out_path + f'/win{win_size}_step{step}_nestedCV_log_{tmp}.log', 'w')

# sys.stdout = log_file
logger.info(t1)

logger.info('Data from all patients with FBTCSs')  # 12 juill 2023

logger.info(f'\nNested cross_validation for Hexoskin data.')
logger.info(f'Features extracted using win_size={win_size} seconds and step={step} seconds')

logger.info(f'There are {len(all_combinations)} combinations using this hyperparameters grid')
logger.info(f'tau_range = {tau_range}')
logger.info(f'thresh_range = {thresh_range}')
logger.info(f'max_depth = {max_depth}')
logger.info(f'min_child_weight = {min_child_weight}')

data = load_data(path=home+'Oumayma/Hexoskin/Features_FBTCS_patients/',
                 win_size=win_size, step=step)


logger.info(f'\nEstimator to be evaluated is {model.__class__.__name__}')

k, inner_k = 7, 6  
# Following the 70-15-15 rule for train-validation-test splits
# 42 patients = 6 test, 6 validation, 30 train
logger.info(f'\n{k} outer folds and {inner_k} inner folds.')

tt1 = datetime.datetime.now()
scores = cross_validate(model, all_combinations, data, k, inner_k,
                        home_path=home)
tt2 = datetime.datetime.now()
logger.info(f'\nTraining time for nested cross-validation is {tt2-tt1}')

logger.info("\n\nMy scores:")
logger.info(scores)

# sys.stdout = old_stdout
# log_file.close()
