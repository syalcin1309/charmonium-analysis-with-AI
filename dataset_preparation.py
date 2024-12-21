import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math  
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

seed = 7 
np.random.seed(seed)

# Set default Seaborn style
#sns.set()


all_pairs_100K = pd.read_csv("Dimuon_DoubleMu_typeG0_T1_massdropped.csv")
all_pairs_100K.shape
all_pairs_100K_withoutna=all_pairs_100K.dropna(axis=0)
all_pairs_100K_withoutna.shape

jpsi_pairs=pd.read_csv("Jpsimumu_typeG0_T1.csv")
jpsi_pairs.shape
jpsi_pairs_withoutna=jpsi_pairs.dropna(axis=0)
jpsi_pairs_withoutna.shape

#adding
dimuon_jpsi = pd.concat([all_pairs_100K_withoutna, jpsi_pairs_withoutna]).drop_duplicates()
dimuon_jpsi.shape
dimuon_jpsi = dimuon_jpsi[dimuon_jpsi['Event'] != 47703178]
dimuon_jpsi.shape

#now calculation mass
dimuon_jpsi['M']=np.sqrt(2*dimuon_jpsi['pt1']*dimuon_jpsi['pt2']*(np.cosh(dimuon_jpsi['eta1']-dimuon_jpsi['eta2'])-np.cos(dimuon_jpsi['phi1']-dimuon_jpsi['phi2'])))
dimuon_jpsi.shape


#adding properties
#checking momentum
#magnitude of p1,p2
dimuon_jpsi['magnitude_p1']=np.sqrt((dimuon_jpsi["px1"]*dimuon_jpsi["px1"])+(dimuon_jpsi["py1"]*dimuon_jpsi["py1"])+(dimuon_jpsi["pz1"]*dimuon_jpsi["pz1"]))
dimuon_jpsi['magnitude_p2']=np.sqrt((dimuon_jpsi["px2"]*dimuon_jpsi["px2"])+(dimuon_jpsi["py2"]*dimuon_jpsi["py2"])+(dimuon_jpsi["pz2"]*dimuon_jpsi["pz2"]))

#adding P columns Px,Py,Pz,P and Pt

dimuon_jpsi["Px"] = dimuon_jpsi["px1"] + dimuon_jpsi["px2"]
dimuon_jpsi["Py"] = dimuon_jpsi["py1"] + dimuon_jpsi["py2"]
dimuon_jpsi["Pz"] = dimuon_jpsi["pz1"] + dimuon_jpsi["pz2"]
dimuon_jpsi['magnitude_P']=np.sqrt((dimuon_jpsi["Px"]*dimuon_jpsi["Px"])+(dimuon_jpsi["Py"]*dimuon_jpsi["Py"])+(dimuon_jpsi["Pz"]*dimuon_jpsi["Pz"]))
dimuon_jpsi['P_t']=np.sqrt((dimuon_jpsi["Px"]*dimuon_jpsi["Px"])+(dimuon_jpsi["Py"]*dimuon_jpsi["Py"]))

dimuon_jpsi.shape


dimuon_jpsi['deltaeta']=np.absolute(dimuon_jpsi['eta1']-dimuon_jpsi['eta2'])
dimuon_jpsi.shape #checking number

#adding phi differences
dimuon_jpsi['deltaphi']=np.absolute(dimuon_jpsi['phi1']-dimuon_jpsi['phi2'])
dimuon_jpsi.shape #checking number



plt.figure(facecolor='white')
plt.hist(dimuon_jpsi['M'],bins=11988,histtype='step',fill=False,lw=2,edgecolor='black')
xlab1 = '$m_{\u03BC\u03BC}$ [GeV/c$^2$]'
ylab1 = 'Events/(0.025 GeV/c$^2$)'
plt.xlabel(xlab1,fontsize=28)
plt.ylabel(ylab1,fontsize=28)
plt.yticks(fontsize=28)
plt.xticks(fontsize=28)
plt.xlim(0.3, 300)
plt.yscale('log')
plt.xscale('log')
plt.show()

#signal definition mass+/-0.3
jpsipairs_fromdimuon=dimuon_jpsi[(dimuon_jpsi['M']>2.7969) &(dimuon_jpsi['M']<3.3969)]
jpsipairs_fromdimuon.shape


signal_jpsi=jpsipairs_fromdimuon.assign(class_label='1')
signal_jpsi.shape


opacity = 0.9
#plotting signals
plt.hist(jpsipairs_fromdimuon["M"],bins=30,histtype='step',fill=True, lw=1.5,color="lightgrey",hatch='',edgecolor="black",alpha=opacity)
xlab1 = 'Invariant Mass [GeV/c$^2$]'
ylab1 = 'Events/(0.02 GeV/c$^2$)'
plt.xlabel(xlab1,fontsize=28)
plt.ylabel(ylab1,fontsize=28)
plt.yticks(fontsize=28)
plt.xticks(fontsize=28)
plt.xlim(2.801,3.3969)
plt.show()


background_dimuon=dimuon_jpsi[(dimuon_jpsi['M']<=2.7969) |(dimuon_jpsi['M']>=3.3969)]
background_dimuon.shape
background=background_dimuon.assign(class_label='0')
background.shape
plt.hist(background_dimuon["M"],bins=297,histtype='step',fill=True,lw=1.5,hatch='',color='darkgrey',edgecolor="black",alpha=opacity) #99
xlab1 = '$m_{\u03BC\u03BC}$ [GeV/c$^2$]'
ylab1 = 'Events/(1.0 GeV/c$^2$)'
plt.xlabel(xlab1,fontsize=28)
plt.ylabel(ylab1,fontsize=28)
plt.yticks(fontsize=28)
plt.xticks(fontsize=28)
plt.xlim(0.3, 300)
plt.show()



#adding them all
signal1=signal_jpsi
signal1.shape
signal_background=signal1.append(background)
signal_background.shape
shuffled_signal_background=signal_background.sample(frac=1)
shuffled_signal_background.shape

shuffled_signal_background = shuffled_signal_background[['class_label','Run','Event','M','eta1','eta2','Q1','Q2','phi1','phi2','px1','px2','py1','py2','pz1','pz2','pt1','pt2','Px','Py','Pz','E1','E2','magnitude_P','P_t','magnitude_p1','magnitude_p2','deltaeta','deltaphi','type1','type2']]

shuffled_signal_background.to_csv('noindex_labeled_shuffled_signal_background.csv', index=False)

#withnoheader
shuffled_signal_background.to_csv('noheader_noindex_labeled_shuffled_signal_background.csv',header=None, index=False)
