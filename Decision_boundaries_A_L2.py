# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:49:15 2022

@author: Juliana RL
"""

import matplotlib.pyplot as plt
import numpy as np
import pybaselines
from pybaselines import utils
import pandas as pd
import os
import glob
import rampy as rp
from scipy.ndimage import uniform_filter1d
import re
from scipy.ndimage import uniform_filter1d
from pybaselines.polynomial import imodpoly, modpoly
from pybaselines.utils import gaussian
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold,cross_val_score

path='J:/Escritorio/Progel/Paper_analysis/treatments_among_gelatins/A/L2' 
csv_files = glob.glob(os.path.join(path, "*.csv"))
# loop over the list of csv files
columns_name=[]
spectra_abs=[]

for f in csv_files:
    head, tail = os.path.split(f)
    spectrum=pd.read_csv(f)
    spectrum.columns=['wave', 'Abs']
    
    #normalización
    spectrum_norm=rp.normalise(spectrum['Abs'],method="intensity")
    x_wave=spectrum['wave']
    y=spectrum_norm
    
    #suavizado
    smooth_y = uniform_filter1d(y, 11)
    
    #correción de linea base
    regular_modpoly = modpoly(y, x_wave, poly_order=3)[0]
    smoothed_modpoly = modpoly(smooth_y, x_wave, poly_order=3)[0]
    
    #sustracción de linea base al espectro
    original_tratado=y-regular_modpoly
    y_sinbase=original_tratado.to_numpy()
    smoothed_tratado=smooth_y-smoothed_modpoly

    df_spectra_tratado=pd.DataFrame(list(zip(x_wave,spectrum['Abs'],y,smooth_y,regular_modpoly,smoothed_modpoly,original_tratado,smoothed_tratado)),
                                         columns=['wave number','original','normalizado','suavizado','linea base(lb) original','lb suavizado','original-lb','suavizado-lb'])
    df_spectra_tratado.to_csv(os.path.join('resumen_tratamiento',tail))
    
    df_spectra_original_tratado=pd.DataFrame(list(zip(x_wave,original_tratado)),columns=['wave number','Abs'])
    df_spectra_original_tratado.to_csv(os.path.join('originales tratados',tail))

##PRE-PROCESSING PLOTS AND START OF PCA/HA ANALYSIS
path2='J:/Escritorio/Progel/Paper_analysis/treatments_among_gelatins/A/L2/resumen_tratamiento' #DIRECTORY OF THE FILE WHERE THE CODE IS SAVE TOGUETHER WITH THE 2 FILES WHERE THE PRETREATMENT RESULTS ARE STORED
csv_files2 = glob.glob(os.path.join(path2, "*.csv"))

#LISTS OF PCA AND HA
columns_name=[]
spectra_abs=[]


for f in csv_files2:
    # read the csv file
    df = pd.read_csv(f)
    df = df.set_index(df.columns[0])
    
    #FOR PCA/HA
    wave_number=df['wave number'].tolist()
    name=re.findall("[A-D][1-3]",f)[-1]
    columns_name.append(name)
    abs_data=df['original-lb'].tolist() 
    spectra_abs.append(abs_data)
    
#DF for PCA/HA ANALYSIS
df_spectra=pd.DataFrame(data=spectra_abs,index=columns_name,columns=wave_number).dropna(axis=1)
wave=df_spectra.columns.values.tolist()
columns_name_df=columns_name.copy()

n_target=[]
for target in columns_name:
    if target in ['A1','B1','C1','D1']:
        n_target.append(0)
    if target in ['A2','B2','C2','D2']:
        n_target.append(1)
    if target in ['A3','B3','C3','D3']:
        n_target.append(2)
        



##### VALIDATION 
#Pre-processing of validation spectra
path_valitadion='J:/Escritorio/Progel/Paper_analysis/treatments_among_gelatins/A/L2/validation' 
csv_files_validation = glob.glob(os.path.join(path_valitadion, "*.csv"))
columns_name=[]
spectra_abs=[]


for f in csv_files_validation:
    head, tail = os.path.split(f)
    spectrum=pd.read_csv(f)
    spectrum.columns=['wave', 'Abs']
    
    #normalización
    spectrum_norm=rp.normalise(spectrum['Abs'],method="intensity")
    x_wave=spectrum['wave']
    y=spectrum_norm
    
    #suavizado
    smooth_y = uniform_filter1d(y, 11)

    #correción de linea base
    regular_modpoly = modpoly(y, x_wave, poly_order=3)[0]

    #sustracción de linea base al espectro
    original_tratado=y-regular_modpoly


    
    df_spectra_validation_t=pd.DataFrame(columns=['wave number','Abs'])
    df_spectra_validation_t['Abs']=original_tratado
    df_spectra_validation_t['wave number']=x_wave
    df_spectra_validation_t.to_csv(os.path.join('validation_pre_process',tail))
    
path3='J:/Escritorio/Progel/Paper_analysis/treatments_among_gelatins/A/L2/validation_pre_process' #DIRECTORY OF THE FILE WHERE THE CODE IS SAVE TOGUETHER WITH THE 2 FILES WHERE THE PRETREATMENT RESULTS ARE STORED
csv_files3 = glob.glob(os.path.join(path3, "*.csv"))

columns_name_val=[]
spectra_abs=[]

for f in csv_files3:
    # read the csv file
    df = pd.read_csv(f)
    df.columns=['index','wave number','original-lb']
    #
    #FOR PCA/HA
    wave_number=df['wave number'].tolist()
    name=re.findall("[A-D][1-3]",f)[-1]
    columns_name_val.append(name)
    abs_data=df['original-lb'].tolist() 
    spectra_abs.append(abs_data)
    
df_spectra_val=pd.DataFrame(data=spectra_abs,index=columns_name_val,columns=wave_number).dropna(axis=1)
wave_val=df_spectra_val.columns.values.tolist() 
  



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
##################################################PCA
pca=PCA(2)
x_pca=pca.fit_transform((scale(df_spectra)))
x_val_pca=pca.transform((scale(df_spectra_val)))

df_x_pca_reduce=pd.DataFrame(data=list(zip(x_pca[:,0],x_pca[:,1],columns_name)), columns=['PC1','PC2','target'])
pc1_variance=float("{:.2f}".format(pca.explained_variance_ratio_[0]))*100
pc2_variance=float("{:.2f}".format(pca.explained_variance_ratio_[1]))*100


X_train,X_test,y_train,y_test = train_test_split(x_pca,n_target,test_size=0.4,random_state=4)
 
# Create color maps
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

n_neighbors = 1
names = [ "NCA, KNN",'SVC','LDA','Decision tree']

###PCA classification models
classifiers = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nca", NeighborhoodComponentsAnalysis()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    ),
    
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ('svc',SVC(kernel='linear'))]),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ('LDA',LDA())]),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ('DT',DecisionTreeClassifier(max_depth=4,random_state=0))])
]

df_predictions_pca=pd.DataFrame()
df_predictions_pca['target']=columns_name_val


cv_scores_pca_train=[]
scores_pca_test=[]
mse_pca=[]
for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores_pca_test.append(score)
    
    mse=mean_squared_error(y_train,clf.predict(X_train))
    mse_pca.append(mse)
    
    
    folds=4
    cv = KFold(n_splits=(folds - 1))
    scores_pca_train = cross_val_score(clf, X_train, y_train, cv = cv)
    cv_scores_pca_train.append(scores_pca_train)
    
    
    
    prediction=clf.predict(x_val_pca)
    model_name='PCA {}'.format(name)
    df_predictions_pca[model_name]=prediction
    
    plt.figure()
    plot_decision_regions(x_pca, np.array(n_target),
                      X_highlight=(X_test),
                      clf=clf,
                      legend=0)
    plt.xlabel('PC1 ({}%)'.format(pc1_variance))
    plt.ylabel('PC2 ({}%)'.format(pc2_variance))
    
    
    c=[]
    for pred in columns_name_val:
        if pred in ['A1','B1','C1','D1']:
            c.append('blue')
        if pred in ['A2','B2','C2','D2']:
            c.append('orange')
        if pred in ['A3','B3','C3','32']:
            c.append('green')
            
    # Plot also the training and testing points
    #plt.scatter(x_val_pca[:, 0],x_val_pca[:, 1],c=c,marker='X', edgecolor="k", s=40)
    #plt.title("PCA-{} (test score:{})".format(name,score))
    plt.legend(loc='best')
    
    plt.show()
    

##################################################PLS
x_pls_=df_spectra.iloc[:,:].values
spectra_Y=df_spectra.index.values



y_pls=[]
for label in spectra_Y:
    if label in['A1','B1','C1','D1']:
        y_pls.append(0)
    if label in ['A2','B2','C2','D2']:
        y_pls.append(1)
    if label in ['A3','B3','C3','D3']:
        y_pls.append(2)

#cross-validation
# 10-fold CV, with shuffle
ten_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
mse_pls = []
for i in np.arange(1, 8):
    pls_ = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls_, x_pls_,y_pls, cv=ten_fold, scoring='neg_mean_squared_error').mean()
    mse_pls.append(-score)
    
pls = PLSRegression(2)
x_pls_uf=pls.fit((scale(x_pls_)),y_pls)
x_pls=pls.transform(scale(x_pls_))
predicted_y=pls.predict(x_pls_)
x_val_pls=pls.transform((scale(df_spectra_val)))

x_train,x_test,Y_train,Y_test = train_test_split(x_pls,y_pls,test_size=0.4,random_state=4) 



df_predictions_pls=pd.DataFrame()

cv_scores_pls_train=[]
scores_pls_test=[]
mse_pls_=[]
for name, clf in zip(names, classifiers):
    clf.fit(x_train, Y_train)
    score = clf.score(x_test, Y_test)
    scores_pls_test.append(score)
    
    mse=mean_squared_error(Y_train,clf.predict(x_train))
    mse_pls_.append(mse)
    
    folds=4
    cv = KFold(n_splits=(folds - 1))
    scores_pls_train = cross_val_score(clf, x_train, Y_train, cv = cv)
    cv_scores_pls_train.append(scores_pls_train)
    

    
    prediction=clf.predict(x_val_pls)
    model_name='PLS {}'.format(name)
    df_predictions_pls[model_name]=prediction
    

    

    plot_decision_regions(x_pls, np.array(y_pls),
                          X_highlight=(x_test),
                          clf=clf,
                          legend=2)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    
    
    c=[]
    for pred in columns_name_val:
        if pred in ['A1','B1','C1','D1']:
            c.append('blue')
        if pred in ['A2','B2','C2','D2']:
            c.append('orange')
        if pred in ['A3','B3','C3','32']:
            c.append('green')
            
    # Plot also the training and testing points
    #plt.scatter(x_val_pls[:, 0],x_val_pls[:, 1],c=c,marker='X', edgecolor="k", s=40)
    #plt.title("PLS-{} (test score:{})".format(name,score))
    plt.legend(loc='best')
    
    plt.show()


############## Figures
#1. Elbow
# Plot cross-validation results 
   
for i in [8]:
    pca_t=PCA(i)
    pca_t.fit_transform((scale(df_spectra)))
    t=pca_t.explained_variance_ratio_
    

fig1, (ax1,ax2)=plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.scatter(np.arange(1, 9),1-t,label='Cummulative explained variance')
ax1.plot(np.arange(1, 9),1-t)
ax1.legend()
#ax1.tick_params(axis='x', which='major', labelsize=0, length=0)
ax1.set_xlabel('Number of PCA Components')

ax2.plot(np.arange(1, 8), np.array(mse_pls),label='mse')
ax2.set_xlabel('Number of PLS Components')
ax2.legend()

c=sns.color_palette('coolwarm', as_cmap=True)
#2.PCA heat map
plt.figure()
map= pd.DataFrame(pca.components_,columns=wave, index=['PC1','PC2'])
sns.heatmap(map,cmap=c).set(title='PCA-LDA')

#3.PLS heat map
pls_loadings=pls.x_loadings_
plt.figure()
map= pd.DataFrame(pls_loadings.T[0:2],columns=wave,index=['LD1','LD2'])
sns.heatmap(map,cmap=c).set(title='PLS-DA')

print('variance PCA:{}'.format (pca.explained_variance_ratio_))
print('Total mse error when using LD1 and LD2:{}'.format (mse_pls[1]))

#7. HCA
plt.figure()
sns.clustermap(df_spectra,method="ward",metric="euclidean",col_cluster=False)
plt.title(label='Dendogram heatmap', fontsize=22)

total=[df_predictions_pca,df_predictions_pls]
z_pred=pd.concat(total, axis=1)
z_pred.set_index('target',inplace=True)

#cajas y bigotes

data_box=[]
for i in cv_scores_pls_train:
    spread=[]
    center=[]
    h=[]
    l=[]
    center.append(i.mean())
    spread.append(np.std(i))
    h.append(i.mean()+np.std(i))
    l.append(i.mean()-np.std(i))
    d_pls=np.concatenate((spread,center,h,l))
    d_pls.shape=(-1,1)
    data_box.append(d_pls.flatten())

for i in cv_scores_pca_train:
    spread=[]
    center=[]
    h=[]
    l=[]
    center.append(i.mean())
    spread.append(np.std(i))
    h.append(i.mean()+np.std(i))
    l.append(i.mean()-np.std(i))
    d_pls=np.concatenate((spread,center,h,l))
    d_pls.shape=(-1,1)
    data_box.append(d_pls.flatten())

names_ = [ "NCA-KNN",'SVC','LDA','DT',
          "NCA-KNN",'SVC','LDA','DT']
fig8 = plt.figure(figsize=(15,10))
ax1 = fig8.add_subplot()
bp=ax1.boxplot(data_box,patch_artist=True)
# fill with colors
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_color('lightblue')
bp['boxes'][1].set_color('lightblue')
bp['boxes'][1].set_facecolor('lightblue')
bp['boxes'][2].set_facecolor('lightblue')
bp['boxes'][2].set_color('lightblue')
bp['boxes'][3].set_color('lightblue')
bp['boxes'][3].set_facecolor('lightblue')

bp['boxes'][4].set_facecolor('indianred')
bp['boxes'][4].set_color('indianred')
bp['boxes'][5].set_color('indianred')
bp['boxes'][5].set_facecolor('indianred')
bp['boxes'][6].set_facecolor('indianred')
bp['boxes'][6].set_color('indianred')
bp['boxes'][7].set_color('indianred')
bp['boxes'][7].set_facecolor('indianred')        
        
#ax1.set_xticklabels(names_,rotation=90,fontsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.set(ylim=(0, 1.5))
ax1.set_ylabel('Crossval train score',fontsize=20)
plt.text(1.8, 1.4, 'PCA', fontsize=28,fontweight="bold",bbox=dict(facecolor='lightblue', alpha=0.5))
plt.text(5.8, 1.4, 'PLS', fontsize=28,fontweight="bold",bbox=dict(facecolor='indianred', alpha=0.5))
plt.text(0.8, 1.2, '(1.00)', fontsize=20)
plt.text(1.8, 1.2, '(1.00)', fontsize=20)
plt.text(2.8, 1.2, '(1.00)', fontsize=20)
plt.text(3.8, 1.2, '(1.00)', fontsize=20)
plt.text(4.8, 1.2, '(1.00)', fontsize=20)
plt.text(5.8, 1.2, '(1.00)', fontsize=20)
plt.text(6.8, 1.2, '(1.00)', fontsize=20)
plt.text(7.8, 1.2, '(1.00)', fontsize=20)
plt.show()

#DENDOGRAM PLOT
import scipy.cluster.hierarchy as sc
 
plt.figure()
plt.tick_params(axis='both', which='major')
dn1=sc.dendrogram(sc.linkage(y=df_spectra, method='ward'),labels=columns_name_df)
#plt.y_label('Dissimilarity')
#DENDOGRAM WITH WARD LINKAGE METHOD
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
plt.tick_params(axis='both', which='major', labelsize=20, length=5)
dn1=sc.dendrogram(sc.linkage(y=df_spectra, method='ward'),labels=columns_name_df)
plt.title(label='Method= Ward', fontsize=22)
#DENDOGRAM WITH SINGLE LINKAGE METHOD
plt.subplot(2,2,2,title='method= Single')
plt.title(label='Method= Single', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=20, length=5)
dn2=sc.dendrogram(sc.linkage(df_spectra, method='single'),labels=columns_name_df)
#DENDOGRAM WITH COMPLETE LINKAGE METHOD
plt.subplot(2,2,3,title='method= Complete')
plt.title(label='Method= Complete', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=20, length=5)
dn3=sc.dendrogram(sc.linkage(df_spectra, method='complete'),labels=columns_name_df)
#DENDOGRAM WITH AVERAGE LINKAGE METHOD
plt.subplot(2,2,4,title='method= Average')
plt.title(label='Method= Average', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=20, length=5)
dn4=sc.dendrogram(sc.linkage(df_spectra, method='average'),labels=columns_name_df)



for col in z_pred.columns:
    column=z_pred[col]
    column.replace(to_replace = 0, value = 'A1', inplace=True)
    column.replace(to_replace = 1, value = 'A2', inplace=True)
    column.replace(to_replace = 2, value = 'A3', inplace=True)