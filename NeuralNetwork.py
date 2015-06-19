# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:14:35 2015

@author: Pavitrakumar
"""

from __future__ import division
import numpy as np
from scipy import optimize

"""
A simple neural netowork with 3 layers.
No. of units in Input layer  -  taken from dataset (X)
No. of units in Hidden layer -  given as user input
No. of units in Output layer -  taken from dataset(y)

So, there will be 2 theta matrices
Theta1 for input->hidden
Theta2 for hidden->output

X is the features matrix - the features
y is the response vector -the output
"""

"""
Size of Theta1 is no.of hiddenlayers(in hidden layer1) x no. of input layers
i.e simply the [A]x[B] where A on the right side corresponds to the no.of units
in the layer on the left, similarly B on the left side(of the dim) corrensponds to
the no.of units in the layer on the right side 
So, our Theta1's dim is hidden_layer_size x input_layer_size + 1 
Similarly the Theta2 it is output_layer_size x hidden_layer_size + 1

NOTE: +1 is for the bias term
"""

"""
Now, we implement the feedforward function to get the cost
cost is h(x) the hypothesis calculated by feeding the input through the
network using the respective weights.
After feed-forward we back propogate to find the delta values,
using this, we find the theta gradients which will be the input to an optimization algorithm
[optimization algorithm used is from scikit]
"""

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g
    

def gradient_of_sigmoid(z):
    sig = sigmoid(z)
    g = sig * (1 - sig)
    return g
    
    
def pack_thetas(Theta1, Theta2):
    #reshaping a matrix here return a vector of size (no. of rows x no. of columns) 
    #i.e each row is appended one after the other to make a huge vector from a matirx
    return np.concatenate((Theta1.reshape(-1), Theta2.reshape(-1)))
        
def unpack_thetas(Theta_combi, input_layer_size, hidden_layer_size, output_layer_size):
    Theta1_start = 0
    Theta1_end = hidden_layer_size * (input_layer_size + 1)
    #Using the total size (which we already know because of the dimensions of the matirx)
    #We can restore the vector into a matrix by simply reshaping it as per its dimensions.
    Theta1 = Theta_combi[Theta1_start:Theta1_end].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = Theta_combi[Theta1_end:].reshape((output_layer_size, hidden_layer_size + 1))
    return Theta1, Theta2
        
def rand_init(l_in, l_out,e):
    #randomly initialize the weights for theta matrices in the range [-e,e]
    return np.random.rand(l_out, l_in) * 2 * e - e
        

def feed_forward_back_prop(Theta1,Theta2,input_layer_size,hidden_layer_size,output_layer_size,X,y,lam,returnJ = False):
    J = 0
    print "Shape of y is: " , np.shape(y),"\n"
    print "Shape of X is: " , np.shape(X),"\n"
    print "output layer size is: " , output_layer_size
    
    Theta1_gradient = np.zeros(np.shape(Theta1))
    Theta2_gradient = np.zeros(np.shape(Theta2))
    #for a multi-layer implementation, take the theta values in a single vector and unroll them   
    
    no_training_samples = np.shape(X)[0]
    
    #since its a classification problem, we build a new Y matrix(binary) from vector of Y
    #i.e it now has output_layer_size columns - each column represents the probability of that class being the result(or output)
    
    new_Y = np.eye(output_layer_size)[y]

    #adding a vector of ones to the X matrix(as the first column) - bias terms for each training example
    X = np.insert(X,0,1,axis=1)
    print "head of X is:\n"
    print X[1:5,:]
    
    #Start the feed-forward steps
    a1 = X
    #a1 = [no. of training examples] x [input_layer_size + 1]
    #theta1 = [hidden_layer_size] x [input_layer_size + 1]
    #theta1 transpose = [input_layer_size + 1] x [hidden_layer_size]
    z2 = np.dot(a1,Theta1.T)
    #z2 = [no. of training examples]  x [hidden_layer_size]
    a2 = sigmoid(z2)
    #For each hidden layer(if it exists), this process continues
    
    #Now, a2 is the new treated as a new X and a similar process to above is carried out
    a2 = np.insert(a2,0,1,axis=1)
    z3 = np.dot(a2,Theta2.T)
    a3 = sigmoid(z3)
    
    #Now we compute the cost function J(theta)
    #f = positive cost - negative cost
    f = np.sum(np.multiply(-new_Y,np.log(a3))-np.multiply((1-new_Y),np.log(1-a3)),axis = 0) # columnwise sum    
    f = (1.0/no_training_samples)*f
    f = sum(f)
    J = f
    print "J is: ",J
    
    reg_term = (lam/(2.0*no_training_samples))*(np.sum(np.sum(np.power(Theta1[:,1:],2),axis=0))+np.sum(np.sum(Theta2[:,1:],axis=0)))
    print "Regularization term is: ",reg_term    
    J = J + reg_term
    print "J after regularization: ",J,"\n"
    
    if returnJ is True:
        return J


    #Now we need to calculate the gradients by backpropagating
    
    #for layer 3 (output layer)
    s_delta_3 = a3-new_Y
    #we are splicing theta2 to get rid of the bias terms
    #for layer 2 (hidden layer)
    s_delta_2 = np.multiply(np.dot(s_delta_3,Theta2[:,1:]),gradient_of_sigmoid(z2))
    #nothing for input layer because we dont aassociate error terms with input layer.
    #now,for accumulating the s_deltas, we use delta
    
    delta_2 = np.dot(a2.T,s_delta_3)
    delta_1 = np.dot(a1.T,s_delta_2)
    
    Theta1_gradient = (1/no_training_samples)*delta_1.T
    Theta2_gradient = (1/no_training_samples)*delta_2.T
    
    #Regularizing the theta terms

    reg_term_T1 = (lam/no_training_samples)*Theta1[:,1:]
    reg_term_T2 = (lam/no_training_samples)*Theta2[:,1:]
    
    Theta1_gradient[:,1:] += reg_term_T1
    Theta2_gradient[:,1:] += reg_term_T2
    
    return Theta1_gradient,Theta2_gradient


def returnsJ(Theta_combi,input_layer_size,hidden_layer_size,output_layer_size,X,y,lam):
    Theta1, Theta2 = unpack_thetas(Theta_combi, input_layer_size, hidden_layer_size, output_layer_size)
    J = feed_forward_back_prop(Theta1,Theta2,input_layer_size,hidden_layer_size,output_layer_size,X,y,lam,True)
    return J
    
def returnThetaGrad(Theta_combi,input_layer_size,hidden_layer_size,output_layer_size,X,y,lam):
    Theta1, Theta2 = unpack_thetas(Theta_combi, input_layer_size, hidden_layer_size, output_layer_size)
    Theta1_grad, Theta2_grad = feed_forward_back_prop(Theta1,Theta2,input_layer_size,hidden_layer_size,output_layer_size,X,y,lam)
    return pack_thetas(Theta1_grad,Theta2_grad)
    

def fit(X,y,maxiter = 750,method = 'TNC',hidden_layer_size = 25,lam = 0):
    input_layer_size = np.shape(X)[1]   #dependant on dataset
    #hidden_layer_size = 50   # user input
    output_layer_size = len(set(y)) 
    
    Theta1 = rand_init(hidden_layer_size, input_layer_size + 1,0.12)
    Theta2 = rand_init(output_layer_size, hidden_layer_size + 1,0.12)
    print "dim of theta1: ",Theta1.shape
    print "dim of theta2: ",Theta2.shape
    Theta_combi = pack_thetas(Theta1,Theta2)
    #Optimization algorithms which returs the optimized values of theta given the gradients and cost function result
    options = {'maxiter': maxiter}
    _res = optimize.minimize(returnsJ, Theta_combi, jac=returnThetaGrad, method=method,args=(input_layer_size, hidden_layer_size, output_layer_size, X, y,lam), options=options)
    t1, t2 = unpack_thetas(_res.x, input_layer_size, hidden_layer_size, output_layer_size)
    return t1,t2
    
def predict(t1,t2,X):
    #here, we just perform a feed-forward step
    h1 = sigmoid(np.dot(np.insert(X,0,1,axis=1),t1.T))
    h2 = sigmoid(np.dot(np.insert(h1,0,1,axis=1),t2.T))
    return h2



"""
testing using IRIS data set
"""

import sklearn.datasets as datasets
from sklearn import cross_validation

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)


#The optimized theta values after training using the training data
t1,t2 = fit(X_train,y_train)


#Predicting using the optimized theta values
pred = predict(t1,t2,X_test)

#predict function returns the matrix of probability of each class for each test data point

pred = pred.argmax(1)

#pred matrix is convertex to a vector where each element represents the index of maximum probability in the row
#i.e it now holds the vector of classes where each element represents the classification result of a test data point


from sklearn.metrics import accuracy_score

"Prediction accuracy: ",print accuracy_score(y_test,pred)
