# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:27:16 2015

@author: Pavitrakumar
"""

from __future__ import division
import numpy as np
from sklearn import datasets
from numpy.linalg import inv
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from scipy import optimize


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g



"""
This function computes the cost (J) of using theta as parameter for 
regularized logistic regression and the gradient (grad) of the cost
w.r.t to the parameters
"""

#if returnJ is True, function returns J 
#if returnJ is False, function returns grad
def cost_function(theta,X,y,lam,returnJ = True):
    #calculating cost
    
    no_of_rows = len(y)
    J = 0

    h = sigmoid(np.dot(X,theta))

    #J = (1.0/no_of_rows)*np.sum(np.multiply(-y,np.log(h))-np.multiply((1-y),np.log(1-h)),axis = 0)
    # both are same! (above and below)
    J = (1.0/no_of_rows)*(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))
    
    reg = (lam/(2.0*no_of_rows))*np.sum(np.power(theta[1:],2))
    J = J + reg

    if returnJ is True:
        return J

    #calculating gradient
    grad = np.zeros(theta.shape)
    error = h - y
    grad[0] = (1.0/no_of_rows)*(np.dot(error.T,X))[0] #for theta0
    grad[1:] = (((1.0/no_of_rows)*(np.dot(error.T,X)).T) + (lam/no_of_rows)*theta)[1:] # for rest of the theta terms except theta0
        
    return grad


def returnJ(theta,X,y,lam):
    return cost_function(theta,X,y,lam)

def returnThetaGrad(theta,X,y,lam):
    return cost_function(theta,X,y,lam,False)




def fit(X,y,maxiter = 50,method = 'TNC',lam = 0.1):
    no_of_rows = X.shape[0]
    no_of_features = X.shape[1]
    no_of_labels = len(set(y))
    fit_theta = np.zeros((no_of_labels,no_of_features+1))
    #adding a vector of ones to the X matrix(as the first column) - bias terms for each training exmaples
    X = np.insert(X,0,1,axis=1)
    
    initial_theta = np.zeros((no_of_features+1,1))

    for i in range(no_of_labels):
        temp_y = (y == (i)) + 0 # here labels are 0,1,2,3.. if they are 1,2,3,4... use: temp_y = (y == (i+1))+0
        #temp_y is a vector of size no_of_training_examples
        #since each iteration corresponds to finding theta for a single class (one-vs-all)
        #each time, we only take the predection of class 'i'on all training example
        
        _res = optimize.fmin_cg(returnJ, fprime=returnThetaGrad,x0 = initial_theta,args=(X, temp_y,lam), maxiter=50, disp=False, full_output=True)
        fit_theta[i,:] = _res[0]  
        """
        different minimization functions (above and below)
        """
        #options = {'maxiter': maxiter}
        #_res = optimize.minimize(returnJ, initial_theta, jac=returnThetaGrad, method=method,args=(X, temp_y,lam), options=options)        
        #fit_theta[i,:] = _res.x

    return fit_theta


def predict(theta,X):
    X = np.insert(X,0,1,axis=1)
    h = np.dot(X,theta.T)
    print h
    return h.argmax(1)


"""
testing using IRIS data set
"""
import sklearn.datasets as datasets
from sklearn import cross_validation


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.9)


#The optimized theta values after training using the training data
theta = fit(X_train,y_train)

    
pred = predict(theta,X_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test,pred)
