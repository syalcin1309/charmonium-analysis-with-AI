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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

#lightgbm model
import lightgbm
print(lightgbm.__version__)
from lightgbm import LGBMClassifier




from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score



import time

from time import process_time
t1_start = process_time() 





# fix random seed for reproducibility
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

#data spliting training and test

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

clf=LGBMClassifier(objective='binary',n_estimators=350,learning_rate=0.2,random_state=47)
#model training

clf = clf.fit(X_train, y_train)


importances = clf.feature_importances_
feature_names = data.columns[data.columns != 'class_label']
# Convert to percentages
total_importance = np.sum(importances)
percentage_importances = (importances / total_importance) * 100

# Sort features based on importance
sorted_idx = np.argsort(percentage_importances)[::-1]



# Print the sorted feature importances
print("Feature Importances:")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {percentage_importances[idx]:.2f}%")



# Plotting
fig = plt.figure(figsize=(10, 7), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

# Set the color of the bars here
bars = plt.bar(np.array(feature_names)[sorted_idx], percentage_importances[sorted_idx], color='steelblue')
plt.ylabel('Importance (%)')
plt.xlabel('Features')
#plt.title('GBDT Feature Importance as Percentage of Total')
plt.xticks(rotation=45, ha='center')

# Adding "GBDT" to the top right
plt.text(0.95, 0.95, 'LightGBM', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


#model evaluation using training and validation sample
# definition of general plotting parameters
nbins = 100

y_train_score = clf.predict_proba(X_train)



def plot_MVAoutput(y_truth, y_score, nbins=100):
    """
    Plots the MVA output as histogram and returns the underlying
    distributions of the positive and the negative class.
    """
    
    discard, y_score_posClass = np.split(y_score,
                                         2,
                                         axis=1)
    
    y_score_posClass_truePos = y_score_posClass[np.array(y_truth==1)]
    y_score_posClass_trueNeg = y_score_posClass[np.array(y_truth==0)]
    
    plt.figure()

    n_total, bins_total, patches_total = \
        plt.hist(y_score[:,1],
                 bins=nbins,
                 alpha=.25,
                 color='black',
                 label='MVA output')
    
    n_trueNeg, bins_trueNeg, patches_trueNeg = \
        plt.hist(y_score_posClass_trueNeg,
                 bins=nbins,
                 alpha=0.5,
                 color='#dd0000',
                 label='true negative')
    
    n_truePos, bins_truePos, patches_truePos = \
        plt.hist(y_score_posClass_truePos,
                 bins=nbins,
                 alpha=0.5,
                 color='green',
                 label='true positive')
    
    plt.title('MVA output distribution (positive class)')
    plt.xlim(-0.05, 1.05)
    plt.xlabel('MVA output')
    plt.ylabel('Entries')
    plt.legend()
    plt.show() 
    return n_truePos, n_trueNeg

n_truePos, n_trueNeg = plot_MVAoutput(y_train, y_train_score, nbins)


#cut efficiencies plot/MVA Cut optimization

MVAcut = np.empty((0))

plt.figure()
fig, ax1 = plt.subplots()
signal_efficiency = np.empty((0))
backgr_efficiency = np.empty((0))
for i in range(nbins):
    signal_efficiency = np.append(signal_efficiency, \
                                  np.sum(n_truePos[i:n_truePos.shape[0]]) / np.sum(n_truePos))
    backgr_efficiency = np.append(backgr_efficiency, \
                                  np.sum(n_trueNeg[i:n_trueNeg.shape[0]]) / np.sum(n_trueNeg))
    MVAcut = np.append(MVAcut, i/(nbins*1.0))
l1 = ax1.plot(MVAcut, signal_efficiency, label='signal efficiency', color='blue')
l2 = ax1.plot(MVAcut, backgr_efficiency, label='background efficiency', color='red')
ax1.set_xlabel('MVA cut')
ax1.set_ylabel('Efficiency')

ax2 = ax1.twinx()
significance_per_MVAcut = np.empty((0))
for i in range(nbins):
    significance_per_MVAcut = np.append(significance_per_MVAcut, \
                                        np.sum(n_truePos[i:n_truePos.shape[0]]) / \
                                        math.sqrt(np.sum(n_truePos[i:n_truePos.shape[0]] + \
                                                         n_trueNeg[i:n_trueNeg.shape[0]])))
    
l3 = ax2.plot(MVAcut, significance_per_MVAcut,
              label='significance',
              color='green')
pos_max = np.argmax(significance_per_MVAcut)
MVAcut_opt = pos_max/(nbins*1.0)
l4 = ax2.plot(pos_max/(nbins*1.0), significance_per_MVAcut[pos_max],
              label='max. significance for cut at %.2f' % MVAcut_opt,
              marker='o', markersize=10, fillstyle='none', mew=2, linestyle='none',
              color='#005500')
ax2.set_ylabel('Significance', color='green')
ax2.tick_params('y', colors='green')

plt.title('MVA cut efficiencies')
lall = l1+l2+l3+l4
labels = [l.get_label() for l in lall]
ax2.legend(lall, labels, loc='lower left')
plt.tight_layout()
plt.show()


#roc curve and overtraining test
def plot_ROCcurve(y_truth, y_score, workingpoint=-1):
    """
    Plots the ROC curve and (if specified) the chosen working point.
    """
    
    fpr, tpr, thresholds = roc_curve(y_truth, y_score[:,1], pos_label=1)
    roc_auc = roc_auc_score(y_truth, y_score[:,1])
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.5f) for GF+CF' % roc_auc)
    #plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f) ' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    
    if workingpoint != -1:
        # find and plot threshold closest to the chosen working point
        close_MVAcut_opt = np.argmin(np.abs(thresholds-workingpoint))
    
       # plt.plot(fpr[close_MVAcut_opt], tpr[close_MVAcut_opt], 'o', markersize=10,
        plt.plot(fpr[close_MVAcut_opt], tpr[close_MVAcut_opt])#,
                
    
    plt.legend(loc=4)
    plt.show()#i added to see plot

plot_ROCcurve(y_train, y_train_score, MVAcut_opt)
#plot_ROCcurve(y_val, y_val_score)



#several more methods for classifier evoluation

#precision recall curve
def plot_precision_recall_curve(y_truth, y_score, workingpoint=-1):
    """
    Plots the precision-recall curve.
    """
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    precision, recall, thresholds_PRC = \
        precision_recall_curve(y_truth,
                               y_score[:,1])
    
    average_precision = average_precision_score(y_truth, y_score[:,1])
    
    plt.figure()
    plt.plot(recall, precision, lw=2,
             label='Precision-recall curve of signal class (area = {1:0.5f})'
                    ''.format(1, average_precision))
    
    if workingpoint != -1:
        # find threshold closest to the chosen working point
        close_optimum = np.argmin(np.abs(thresholds_PRC-workingpoint))
        
        plt.plot(recall[close_optimum], precision[close_optimum],
                 'o',
                 markersize=10,
                 label="threshold at %.5f" % workingpoint,
                 fillstyle="none",
                 mew=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'Recall $R=T_p / (T_p+F_n)$')
    plt.ylabel(r'Precision $P=T_p / (T_p+F_p)$')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_precision_recall_curve(y_train, y_train_score, MVAcut_opt)


#several more methods for classifier evoluation

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



# Compute confusion matrix
y_train_score_labels = (y_train_score[:,1] > MVAcut_opt)
cnf_matrix = confusion_matrix(y_train, y_train_score_labels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['negative class','positive class'],
                      title='Confusion matrix (non-normalized)')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['negative class','positive class'],
                      normalize=True,
                      title='Normalized confusion matrix')


#classification report
print(classification_report(y_train, y_train_score_labels,
                            target_names=['negative class','positive class'],digits=5))


#model application on test sample

y_test_score = clf.predict_proba(X_test)

n_truePos, n_trueNeg = plot_MVAoutput(y_test, y_test_score)


#ROC Curve

plot_ROCcurve(y_test, y_test_score, MVAcut_opt)

#precision curve

plot_precision_recall_curve(y_test, y_test_score, MVAcut_opt)

# Compute confusion matrix
y_test_score_labels = (y_test_score[:,1] > MVAcut_opt)
cnf_matrix = confusion_matrix(y_test, y_test_score_labels)
np.set_printoptions(precision=5)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['negative class','positive class'],
                      normalize=False,
                      title='Confusion matrix')


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['negative class','positive class'],
                      normalize=True,
                      title='Normalized confusion matrix')

#classification report of test data
print(classification_report(y_test, y_test_score_labels,
                            target_names=['negative class','positive class'],digits=5))

print("Accuracy on test set:   {:.5f}".format(clf.score(X_test, y_test)))


print('*****************')
#gmean
gmean_train = geometric_mean_score(y_train, y_train_score_labels)
print("Geometric Mean for Train:", gmean_train)

gmean_test = geometric_mean_score(y_test, y_test_score_labels)
print("Geometric Mean for Test:", gmean_test)



#bacc
bacc_train = balanced_accuracy_score(y_train, y_train_score_labels)
print("Balanced Accuracy for Train:", bacc_train)

bacc_test = balanced_accuracy_score(y_test, y_test_score_labels)
print("Balanced Accuracy for Test:", bacc_test)



# Stop the stopwatch / counter
t1_stop = process_time()
   
print("Elapsed time:", t1_stop, t1_start) 
   
print("Elapsed time during the whole program in seconds:",
                                         t1_stop-t1_start) 





