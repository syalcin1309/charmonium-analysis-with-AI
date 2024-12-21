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
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold, cross_validate


from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score


#dnn stteings
#import tensorflow as tf
import keras.backend as K
#print(K.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

sess = tf.Session()
K.set_session(sess)

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.utils import class_weight
##########

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.utils import class_weight


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

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



print('\nTotal number of events in data sample: %d' % X.shape[0])
print('Number of signal events in data sample: %d (%.2f percent)' % (y[y==1].shape[0], y[y==1].shape[0]*100/y.shape[0]))
print('Number of backgr events in data sample: %d (%.2f percent)' % (y[y==0].shape[0], y[y==0].shape[0]*100/y.shape[0]))






def create_model(input_dim):
    clf = Sequential()
    clf.add(Dense(20, activation='relu', input_dim=input_dim))
    clf.add(Dense(145, activation='relu'))
    clf.add(Dropout(0.1))
    clf.add(Dense(20, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return clf


# K-Fold cross-validation setup

# K-fold cross-validation
kf= StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Arrays to store metrics
precisions = []
recalls = []
f1s = []
gmeans = []
balanced_accs = []

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Compute class weights for current split
    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(weights))

    # Create and compile the model
    model = create_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=75, batch_size=512, class_weight=class_weights, verbose=1)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate metrics
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1s.append(f1_score(y_test, y_pred, average='weighted'))
    gmeans.append(geometric_mean_score(y_test, y_pred, average='weighted'))
    balanced_accs.append(balanced_accuracy_score(y_test, y_pred))

# Calculate and print average and standard deviation of the metrics
print(f"Weighted Precision - Mean: {np.mean(precisions):.5f}, Std: {np.std(precisions):.5f}")
print(f"Weighted Recall - Mean: {np.mean(recalls):.5f}, Std: {np.std(recalls):.5f}")
print(f"Weighted F1-Score - Mean: {np.mean(f1s):.5f}, Std: {np.std(f1s):.5f}")
print(f"Geometric Mean - Mean: {np.mean(gmeans):.5f}, Std: {np.std(gmeans):.5f}")
print(f"Balanced Accuracy - Mean: {np.mean(balanced_accs):.5f}, Std: {np.std(balanced_accs):.5f}")


