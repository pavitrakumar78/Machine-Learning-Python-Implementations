# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:08:41 2015

@author: Pavitrakumar
"""
from __future__ import division
import numpy as np
from sklearn import datasets
from numpy.linalg import inv
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error



#We use gradient descent to learn the theta values for linear regression
#Theta is the main parameter of our model

#The dimensions of theta matrix is [no. of features] x 1
#No. of features is the no. of columns in the X matrix

#One way to solve for theta

def normal_eqn_theta(X,y):
    #This is the closed-form solution to lienar regression
    
    #Insert a column(axis = 1) of 1s at 0th pos.
    X = np.insert(X,0,1,axis=1)
    
    a = inv(np.dot(X.T , X))
    b = X.T
    c = y
    
    theta = np.dot(np.dot(a,b),c)
    
    #theta = y * X.T * inv(X*X.T)
    #return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta



#Another way to solve for theta is to use gradient descent
#For this method, we need to normalize our input matrix (X)


def gradient_descent_theta(X,y,alpha = 1e-7,num_iters = 500):
    #X - we need to normalize and add bias terms
    #print X[1:5,:],"\n\n"
    X = normalize(X)
    X = np.insert(X,0,1,axis=1)
    theta = np.zeros((X.shape[1],1))
    #print X[1:5,:]
    #Alpha is the learning rate
    #num_iters is the total number of gradient steps
    print "X shape is ",X.shape
    print "Theta shape is(b) ",theta.shape
    
    no_of_rows = len(y) # number of training examples
    J_history = np.zeros((num_iters,1))
    #We can make use of the J-History vector to visualize how the cost is minimized
    y = np.asmatrix(y)
    y = y.T
    for i in range(num_iters):
        h = np.dot(X,theta)
        error = h - y
        gradient = np.dot(error.T,X).T
        theta_change = (alpha/no_of_rows)*gradient
        theta = theta - theta_change
        J_history[i] = compute_cost(X,y,theta)
    print "Theta shape is(a) ",theta.shape
    return theta,J_history
    
def compute_cost(X,y,theta):
    #computes the cost of theta as parameter for linear regression to fit the 
    #data point in X and y
    no_of_rows = len(y) # number of training examples
    J = 0 # J is the cost
    h = np.dot(X,theta)
    square_error = np.power((h-y),2)
    J = (1.0/(2.0*no_of_rows))*np.sum(square_error)
    
    return J
        
def normalize(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    """
    First, for each feature dimension, compute the mean
    of the feature and subtract it from the dataset,
    storing the mean value in mu. Next, compute the 
    standard deviation of each feature and divide
    each feature by it's standard deviation, storing
    the standard deviation in sigma. 
    
    Note that X is a matrix where each column is a 
    feature and each row is an example. You need 
    to perform the normalization separately for 
    each feature. - taken from Andrew Ng's comments
    """
    mu = np.mean(X,axis = 0)
    #Taking column-wise mean
    X_norm = X_norm - mu
    sigma = np.std(X,axis = 0)
    X_norm = X_norm/sigma
    
    return X_norm
    


def lin_reg(X,theta): 
    X = np.insert(X,0,1,axis=1)
    theta = np.asmatrix(theta)
    print "LR-theta shape",theta.shape
    print "LR-X shape",X.shape
    pred = np.dot(X,theta)
    return pred




"""
Loading and training on toy dataset (boston land prices)
"""

boston = datasets.load_boston()

"""
linear regression with multiple variables
"""
X = boston.data
y = boston.target

"""
#linear regression with single variable
X = np.asmatrix(boston.data[:,0]).T #taking only the 1st column
y = boston.target
"""


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)



normal_eqn_theta = np.asmatrix(normal_eqn_theta(X_train,y_train)).T


gradient_descent_theta,_ = gradient_descent_theta(X_train,y_train)




pred1 = lin_reg(X_test,normal_eqn_theta)
pred2 = lin_reg(X_test,gradient_descent_theta)



print "MSE for prediction using normal_eqn Theta is: ", mean_squared_error(y_test, pred1)  
print "MSE for prediction using gradient_desc Theta is: ", mean_squared_error(y_test, pred2)  
