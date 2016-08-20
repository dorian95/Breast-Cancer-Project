# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 15:25:34 2016

@author: Nursultan
"""

from __future__ import print_function
import scipy
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.cross_validation import cross_val_score

sns.set(color_codes= True)
path = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
#path = "C:/Users/Nursultan/Document/Data Science/breast cancer/wdbc.data.csv/"
df = pd.read_csv(path, sep=',', header = None)
df.columns  = ['ID', 'Diagnosis',
                    'MRadius', 'MTexture', 'Mper-ter', 'Marea', 'MSmooth-es',
                    'MComp-ness', 'MConcavity', 'MConcavPoints', 'MSymm-ry',
                    'MFractDimens', 'RadiusSE', 'TextureSE', 'Per-terSE',
                    'AreaSE', 'Smooth-esSE',
                    'Comp-nessSE', 'ConcavitySE', 'ConcavPointsSE', 
                    'Symm-rySE', 'FractDimensSE', 'WRadius', 'WTexture',
                    'WPer-ter', 'WArea', 'WSmooth-es',
                    'WComp-ness', 'WConcavity', 'WConcavPoints', 
                    'WSymm-ry', 'WFractDimens']
df['Malignant'] = df.Diagnosis.map({'B':0, 'M':1})

feature_cols = ['MRadius', 'MTexture', 'Mper-ter', 'Marea', 'MSmooth-es',
                    'MComp-ness', 'MConcavity', 'MConcavPoints', 'MSymm-ry',
                    'MFractDimens', 'RadiusSE', 'TextureSE', 'Per-terSE',
                    'AreaSE', 'Smooth-esSE',
                    'Comp-nessSE', 'ConcavitySE', 'ConcavPointsSE', 
                    'Symm-rySE', 'FractDimensSE', 'WRadius', 'WTexture',
                    'WPer-ter', 'WArea', 'WSmooth-es',
                    'WComp-ness', 'WConcavity', 'WConcavPoints', 
                    'WSymm-ry', 'WFractDimens']
X = df[feature_cols]
#target_names = df.Diagnosis
y = df.Malignant
pca3 = decomposition.PCA(n_components=3)
X_trf_3 = pca3.fit_transform(X)
#print(X_trf_3[0:5])
print(pca3.explained_variance_ratio_.sum())

pca_all = decomposition.PCA()
X_trf_all = pca_all.fit_transform(X)
print(pca_all.explained_variance_ratio_.sum())

plt.cla()
plt.plot(pca_all.explained_variance_ratio_)
plt.title('Variance explained by each principal component\n')
plt.ylabel(' % Variance explained')
plt.xlabel('Principal component')
plt.show()
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
log_reg = logistic.fit(X,y)
scores = cross_val_score(log_reg, X, y, cv=10, scoring ='accuracy')
print(scores.mean())

from sklearn.pipeline import Pipeline
pipe_trf_3 = Pipeline([('pca_3', decomposition.PCA(n_components=3)),
                       ('logistic', LogisticRegression())])
scores_trf_3 = cross_val_score(pipe_trf_3, X, y, cv=10, scoring='accuracy')
print(scores_trf_3.mean())

X_reconstituted = pca3.inverse_transform(X_trf_3)
scores_trf_recon = cross_val_score(log_reg, X_reconstituted, y, cv=10,
                                   scoring='accuracy')
print(scores_trf_recon.mean())

sns.lmplot(x='WComp-ness', y='Malignant', data = df, logistic=True, y_jitter=.03)
sns.lmplot(x='WArea', y='Malignant', data = df, logistic=True, y_jitter=.03)
sns.lmplot(x='MTexture', y='Malignant', data = df, logistic = True, y_jitter=.03)
sns.lmplot(x='WSmooth-es', y='Malignant', data = df, logistic = True, y_jitter=.03)

Means = ['MRadius', 'MTexture', 'Mper-ter', 'Marea', 'MSmooth-es',
                    'MComp-ness', 'MConcavity', 'MConcavPoints', 'MSymm-ry',
                    'MFractDimens']
StandErr = ['RadiusSE', 'TextureSE', 'Per-terSE',
                    'AreaSE', 'Smooth-esSE',
                    'Comp-nessSE', 'ConcavitySE', 'ConcavPointsSE', 
                    'Symm-rySE', 'FractDimensSE']
Worsts = ['WRadius', 'WTexture',
                    'WPer-ter', 'WArea', 'WSmooth-es',
                    'WComp-ness', 'WConcavity', 'WConcavPoints', 
                    'WSymm-ry', 'WFractDimens']
                    
sns.heatmap(df[Means].corr())
sns.heatmap(df[StandErr].corr())
sns.heatmap(df[Worsts].corr())

