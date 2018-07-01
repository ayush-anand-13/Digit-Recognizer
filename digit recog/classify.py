#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 05:11:09 2018
Digit Classifier
@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataset = pd.read_csv('train.csv')
y = dataset.iloc[:,0].values
X = dataset.iloc[:,1:786].values

#convert into y
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 393, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784,))
    classifier.add(Dense(units = 393, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 50,verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
accuracies = cross_val_score(estimator = classifier, X = X, y = dummy_y, cv = kfold, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

classifier2 = Sequential()
classifier2.add(Dense(units = 393, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784,))
classifier2.add(Dense(units = 393, kernel_initializer = 'uniform', activation = 'relu'))
classifier2.add(Dense(units = 190, kernel_initializer = 'uniform', activation = 'relu'))
classifier2.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))
classifier2.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier2.fit(X, dummy_y, batch_size = 40, epochs = 200)

dataset2 = pd.read_csv('test.csv')
X_test = dataset2.iloc[:,:].values

X_test = sc.transform(X_test)

y_pred = classifier2.predict(X_test)
ans = np.argmax(y_pred, axis=1)

b = range(1,28001)

pos =  np.concatenate((b,ans ), axis=0)
pos = pos.reshape(2,28000)
pos = pos.T

np.savetxt("foo8.csv", pos, delimiter=",")


3.8 loss
