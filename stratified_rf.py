import math
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix, radviz 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score


from sklearn.model_selection import StratifiedKFold, cross_validate


seed = 7
np.random.seed(seed)

attributes = [
      'class_label', # 1 for signal, 0 for background
    ##########################################################
    'Run', 
    'Event',
    'M',
    '$\eta_1$',
    '$\eta_2$',
    '$Q_1$',
    '$Q_2$',
    '$\phi_1$',
    '$\phi_2$',
    'p$_{x1}$',
    'p$_{x2}$',
    'p$_{y1}$',
    'p$_{y2}$',
    'p$_{z1}$',
    'p$_{z2}$',
    'p$_{T1}$',
    'p$_{T2}$',
    '$P_x$',
    '$P_y$',
    '$P_z$',
    '$E_1$',
    '$E_2$',
    ########################################################
    'P',
    'P$_{T}$',
    '$p_1$',
    '$p_2$',
    '$\Delta\eta$',
    '$\Delta\phi$',
    'type1',
    'type2'
]


attributes = [
    'class_label', # 1 for signal, 0 for background
    ##########################################################
    'Run', 
    'Event',
    'M',
    '$\eta_1$',
    '$\eta_2$',
    '$Q_1$',
    '$Q_2$',
    '$\phi_1$',
    '$\phi_2$',
    'p$_{x1}$',
    'p$_{x2}$',
    'p$_{y1}$',
    'p$_{y2}$',
    'p$_{z1}$',
    'p$_{z2}$',
    'p$_{T1}$',
    'p$_{T2}$',
    '$P_x$',
    '$P_y$',
    '$P_z$',
    '$E_1$',
    '$E_2$',
    ########################################################
    'P',
    'P$_{T}$',
    '$p_1$',
    '$p_2$',
    '$\Delta\eta$',
    '$\Delta\phi$',
    'type1',
    'type2'
]

#using code for halfsignalfullbg
data = pd.read_csv('noheader_noindex_labeled_shuffled_signal_background.csv',
                   header=None,
                   sep=',',
                   names=attributes,
                   usecols=[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,27,28],
                   nrows=110000 
                  )



X = data.drop(['class_label'], axis=1)
y = (data['class_label']).astype(int)

print('Dimensions of feature matrix X: ', X.shape)
print('Dimensions of target vector y:  ', y.shape)

#if the data is imbalances or balances : same number of signal bg
#in here 52% signal 47% bg

print('\nTotal number of events in data sample: %d' % X.shape[0])
print('Number of signal events in data sample: %d (%.2f percent)' % (y[y==1].shape[0], y[y==1].shape[0]*100/y.shape[0]))
print('Number of backgr events in data sample: %d (%.2f percent)' % (y[y==0].shape[0], y[y==0].shape[0]*100/y.shape[0]))

#data visualization
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=None, 
                             min_samples_leaf=1,
                             random_state=39,
                             verbose=0)
                             



skfold = StratifiedKFold(n_splits=10)

# Scoring metrics
scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'gmean': make_scorer(geometric_mean_score, average='weighted'),
    'balanced_acc': 'balanced_accuracy'
}

# Cross-validation scores
scores = cross_validate(clf, X, y, scoring=scoring, cv=skfold, n_jobs=-1)

# Displaying results
for metric in scores:
    if metric.startswith('test_'):
        metric_name = metric[5:]
        avg_score = np.mean(scores[metric])
        std_dev = np.std(scores[metric])
        print(f"{metric_name.capitalize()} - Mean: {avg_score:.5f}, Std Dev: {std_dev:.5f}")
