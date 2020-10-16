# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:17:04 2020

@author: Priya Bhatia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("C:/Users/Priya Bhatia/Desktop/AI Study Material/Foundations of Machine Learning/K_Nearest_Neighobor/Classified Data")
dataset.head

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset.drop('TARGET CLASS',axis = 1))

#Independent Features
scaled_features = scaler.transform(dataset.drop('TARGET CLASS',axis = 1))

dataset_feat = pd.DataFrame(scaled_features,columns=dataset.columns[:-1])
dataset_feat.head

#Training Testing Split of Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_features,dataset['TARGET CLASS'],test_size = 0.30)

#K-Nearest Neighobor
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 1)
KNN.fit(X_train,y_train)

pred = KNN.predict(X_test)

#Predictions and Evaluations

from sklearn.metrics import confusion_matrix, classification_report
results = confusion_matrix(y_test,pred)
results_report = classification_report(y_test,pred)


#Choosing best K value
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    prediction_i = knn.predict(X_test)
    error_rate.append(np.mean(prediction_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color = 'blue',linestyle = 'dashed',marker = 'o',markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')







