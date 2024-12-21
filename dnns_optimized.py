import math
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix, radviz #not tools module changed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



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




from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score
import time
#startTime = time.time()

from time import process_time
t1_start = process_time() 



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

#datapreprocessing preparing data before start to use in the analysis
scaler = StandardScaler()

# first finds the correct values for the mean and the standard deviation ("fit") and
# then transforms the data using these values ("transform")
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)



print('Number of training samples:   %d' % X_train.shape[0])
#print('Number of validation samples: %d' % X_val.shape[0])
print('Number of test samples:       %d' % X_test.shape[0])

#model accuracy
def get_model(trainX, trainy):
    
    clf = Sequential()
    clf.add(Dense(20,   activation='relu',input_dim=X_train.shape[1]))#was 100
   
    
    #clf.add(Dropout(0.1))
  
    
    clf.add(Dense(145,   activation='relu'))
  
    clf.add(Dropout(0.1))
    
    
       
    
    clf.add(Dense(20,   activation='relu'))#I added

   
    
    clf.add(Dense(1, activation='sigmoid'))

    
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    

    
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train[:])

    # Train the clf
    clf.fit(trainX,
            trainy,
            epochs=75,
            batch_size=512,
            shuffle=True,
            verbose=1)

    return clf

model = get_model(X_train, y_train)



def permutation_feature_importance(model, X, y_true):
    baseline_accuracy = accuracy_score(y_true, np.round(model.predict(X)))
    feature_importances = []

    for i in range(X.shape[1]):
        X_copy = X.copy()
        np.random.shuffle(X_copy[:, i])
        shuffled_accuracy = accuracy_score(y_true, np.round(model.predict(X_copy)))
        importance = baseline_accuracy - shuffled_accuracy
        feature_importances.append(importance)

    return np.array(feature_importances)

# Replace with your actual training data and feature names
feature_names =  data.columns[data.columns != 'class_label']

importances = permutation_feature_importance(model, X_train, y_train)

# Calculate the percentage importance
total_importance = np.sum(np.abs(importances))
percentage_importances = (np.abs(importances) / total_importance) * 100

# Sort features based on importance
sorted_idx = np.argsort(percentage_importances)[::-1]

# Print and plot
print("Feature Importance as Percentage of Total:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {percentage_importances[idx]:.2f}%")

plt.figure(figsize=(10, 7))
plt.bar(np.array(feature_names)[sorted_idx], percentage_importances[sorted_idx],color='steelblue')
plt.ylabel('Importance (%)')
plt.xlabel('Features')
#plt.title('Feature Importance as Percentage of Total')
plt.xticks(rotation=45, ha='center')
plt.tight_layout()

plt.text(0.95, 0.95, 'DNN', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')


plt.show()
############################3


#test_loss, test_acc = clf.evaluate(X_test, y_test)
train_loss,train_acc=model.evaluate(X_train, y_train)
print('Train accuracy:', train_acc)
print('Train loss:', train_loss)

#this part is question
test_loss,test_acc=model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#for training see the classification report
ytrain_probs = model.predict(X_train)
ytrain_classes=(model.predict(X_train) > 0.5).astype("int32")

#adding new lines to make them 1d array
ytrain_probs_1D = ytrain_probs[:, 0]
ytrain_classes_1D = ytrain_classes[:, 0]


#classification report of test data
print(classification_report(y_train, ytrain_classes_1D,
                            target_names=['negative class','positive class'],digits=5))

#confusion matrix for train
#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plots the confusion matrix. If 'normalize' is set 'True',
    the output matrix will contain percentages instead of
    absolute numbers.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



#confusion matrix for training
cnf_matrix_train = confusion_matrix(y_train, ytrain_classes_1D)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_train,
                      classes=['negative class','positive class'],
                      title='Confusion matrix (non-normalized)')




#test
fpr_dnn_train, tpr_dnn_train, thresholds_dnn_train  = roc_curve(y_train, ytrain_probs)
#print('thresholds %.5f' ,thresholds_dnn_train)
roc_auc_dnn_train = auc(fpr_dnn_train, tpr_dnn_train)


plt.plot(fpr_dnn_train, tpr_dnn_train, 'b', label='Simple DNN_TRAIN classifier, AUC = %0.5f'% roc_auc_dnn_train)

plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc = 'lower right')
plt.title("Receiver operating characteristic")
plt.ylabel('TPR')
plt.xlabel('FPR');



plt.show()
auc_score=roc_auc_score(y_train, ytrain_probs)
print('train auc', auc_score)



ypred_probs = model.predict(X_test)#, verbose=0)
# predict crisp classes for test set
ypred_classes=(model.predict(X_test) > 0.5).astype("int32") #round yapak gerekli mi 0.5 ustu hersey denilmis 
#np.argmax(ypred_probs,axis=1)



#adding new lines to make them 1d array
ypred_probs_1D = ypred_probs[:, 0]
ypred_classes_1D = ypred_classes[:, 0]

#classification report of test data
print(classification_report(y_test, ypred_classes_1D,
                            target_names=['negative class','positive class'],digits=5))



#confusion matrix for training
cnf_matrix_test= confusion_matrix(y_test, ypred_classes_1D)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_test,
                      classes=['negative class','positive class'],
                      title='Confusion matrix (non-normalized)')




#roc part
#test
fpr_dnn, tpr_dnn, thresholds_dnn  = roc_curve(y_test, ypred_probs)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)


plt.plot(fpr_dnn, tpr_dnn, 'b', label='Simple DNN classifier, AUC = %0.5f'% roc_auc_dnn)

plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc = 'lower right')
plt.title("Receiver operating characteristic")
plt.ylabel('TPR')
plt.xlabel('FPR');



plt.show()
auc_score_test=roc_auc_score(y_test, ypred_probs)
print('test auc', auc_score_test)

#avg_precision = average_precision_score(y_test, ypred_probs, pos_label=1)
#print(avg_precision)



#gmean
gmean_train = geometric_mean_score(y_train, ytrain_classes_1D)
print("Geometric Mean for Train:", gmean_train)

gmean_test = geometric_mean_score(y_test, ypred_classes_1D)
print("Geometric Mean for Test:", gmean_test)



#bacc
bacc_train = balanced_accuracy_score(y_train, ytrain_classes_1D)
print("Balanced Accuracy for Train:", bacc_train)

bacc_test = balanced_accuracy_score(y_test, ypred_classes_1D)
print("Balanced Accuracy for Test:", bacc_test)






# Stop the stopwatch / counter
t1_stop = process_time()
   
print("Elapsed time:", t1_stop, t1_start) 
   
print("Elapsed time during the whole program in seconds:",
                                         t1_stop-t1_start) 

