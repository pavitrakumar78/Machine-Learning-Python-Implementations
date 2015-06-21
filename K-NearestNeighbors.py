# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:06:04 2015

@author: Pavitrakumar
Credits: Jason Brownlee[Machinelearningmastery.com]
"""

from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import math
import operator

"""
Euclidean distance measure: This is defined as the square root of the sum of the 
squared differences between the two arrays of numbers
"""

def euclideanDistance(instance1, instance2, no_of_features):
    distance = 0
    for x in range(no_of_features):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

"""
getNeighbors function returns k most similar neighbors from the training set 
for a given test instance (using the already defined euclideanDistance function)
"""
def getNeighbors(X_train,y_train, test_instance, k):
    # getting the k-nearest neighbors of the data point testInsatance
    distances = []
    no_of_features = len(test_instance)
    for x,y in zip(X_train,y_train): 
        # we are finding distance from each training example to out testInstance data point
        # and storing it as a list of pairs i.e (ith training example's response,distance to our instance data point)
        dist = euclideanDistance(test_instance, x, no_of_features)
        distances.append((y, dist)) 
    distances.sort(key=operator.itemgetter(1))
    #sorting the list by the 2nd element in each pair - sorting by distance
    #extracting the top k elements from the sorted list
    #we only need the response
    neighbors = [response for (response,distance) in distances]
    neighbors = neighbors[0:k]
    return neighbors
"""
getReponse just returns the most commonly occuring class in the given set of neighbors
"""
def getResponse(neighbors):
    # neighbors is a vector of length k 
    # now, all we need to do is to find the most occuring class
    counts = np.bincount(neighbors)
    max_count = np.argmax(counts)
    return max_count

def predict(X_test,X_train,y_train,k = 5):
    predicted = []
    for each_test_instance in X_test:
        neighbors = getNeighbors(X_train,y_train,each_test_instance,k)
        predicted.append(getResponse(neighbors))
    return predicted
    

"""
testing using IRIS data set
"""

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.8)


pred = predict(X_test,X_train,y_train)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test,pred)
