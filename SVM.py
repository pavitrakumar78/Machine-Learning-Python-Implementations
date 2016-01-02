# -*- coding: utf-8 -*-
"""
Created on Sat Jan 02 12:40:14 2016

@author: Pavitrakumar
"""

from __future__ import division
import numpy as np
from sklearn import datasets
import cvxopt
import cvxopt.solvers
from sklearn.cross_validation import train_test_split

#generating linearly seprable data set.
#Target values = -1 or 1.

mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov = np.array([[0.8, 0.6], [0.6, 0.8]])
X1 = np.random.multivariate_normal(mean1, cov, 100)
y1 = np.ones(len(X1))
X2 = np.random.multivariate_normal(mean2, cov, 100)
y2 = np.ones(len(X2)) * -1
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#X is 150x 4 matrix (2 featues, 150 data points)
#y is 150x 1 matrix 

n_samples = X_train.shape[0]
n_features = X_train.shape[1]

def kernel(x,y):
    return np.dot(x,y)

K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i,j] = kernel(X_train[i], X_train[j])
        
#K is (150,150) matrix
        
#look at lecture notes, each term of P matrix is like y1y1(X1^T*X1)...
#we have already calculated the (Xi^T*Xj) part in matix K.
#now we need to calculate the yiyj part - since yi is just scalar, we just compute the outer product.

Y = np.outer(y_train,y_train)

#now to combine K and Y into P

P = cvxopt.matrix(Y * K)


#the quadratic equation we are trying to minimize is :
# 1/2 x^T*P*x + Q^T*X
#we have computed P already, now for Q:

#according to the final minimization equation equation, the co-efficient for sigma xn is (-1)
#so, we pass a col-matrix of -1s as Q

Q = cvxopt.matrix(-1 * np.ones(n_samples))


#now, we have the solver which looks like this:
#cvxopt.solvers.qp(P, Q, G, h, A, b) 

#We also need to pass in the constraints h,A,b,G



A = cvxopt.matrix(y_train, (1,n_samples),tc='d') #reshaping (1x150 matrix)
b = cvxopt.matrix(0.0)

#The above A and b correspnd to Ax=b general form and y^T*x=0 as a constraint in our equation.

#for general Qp using cvxopt refer: #https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

#G and h general form: Gx <= h and 0<=xn<=C as a constraint in our equation if we need margin. For non-margins, we have 0 <= xn.

#for non-margin support vectors, 0 <= xn but, it is not in the form Gx <= h. To convert it to that form, we have to convert it as:
# (-1)* xn <= 0
# so, G will be a matrix of diaognals -1 and h will be a 150x1 matrix of 0s
# RHS = 0 so h is a 1d matrix of 0 as coefficient is easy to understand.
# Why does the G matrix have a diaognal of -1s?
# Since our goal is to find the "support vectors" and these support vectors are a subset of points from the exisitng data point.
# (support vectors define the margin of the hyperplane we are trying to find out)
# So, each of the data point is to be considered as a unknown variable and the resultant value decides whether it is a support vector or not.
#If we have 5 data points, equations are: 1*x1+0*x2+0*x3+0*x4+0*x5 = 0 so here h = [0] and G = [1,0,0,0,0] .... 0*x1+0*x2+0*x3+0*x4+1*x5 = 0 so here h = [0] and G = [0,0,0,0,1]
G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1)) # 150x150
h = cvxopt.matrix(np.zeros(n_samples)) # 150x1
   
#for margin support vectors, we have a range in which xn lies i.e 0<=xn<=C
#similar to previous where we had only one equality constraint to take care of, here we have 2.
#that is: 0<=xn<=C can be split in 2. (we need to split because constraints to QP algo has to be in the form Gx<=h)
# 0 < xn ==> (-1)*xn < = 0   (is in the form Gx<=h) .....(1)
# (1)*xn <= C ==> (1)*xn <= C   (is in the form Gx<=h, no need to change signs) ...........(2)
   
   

#So, the matrix for G is the previous matrix stacked(vertically) with another diaognal matrix of 1s

tmp1 = np.diag(np.ones(n_samples) * -1) # = G if non-margin SVs (only using (1))
tmp2 = np.identity(n_samples) # using (2) (1 x xn each row)
G = cvxopt.matrix(np.vstack((tmp1, tmp2))) # vertically stack the 2 matrices above
#G is now a 300x150 matrix

#Similarly, for h here xn <= C so h matrix also has to take care of that constraint 
#It is a stacked(horizontally) with a (1x150 stacked horizontally with 1x150) 1x150 matrix of C value.

C = 2
tmp1 = np.zeros(n_samples) # = h if non-margin SVs (only using (1))
tmp2 = np.ones(n_samples) * C # using (2)
h = cvxopt.matrix(np.hstack((tmp1, tmp2))) # 300x1 matrix
#h is 1x300 matrix [0,0,0,....0,0,C,C,....,C,C,C]




#Now we have all the required variables needed to pass to the QP solver and get the xns (the support vectors)

solution = cvxopt.solvers.qp(P, Q, G, h, A, b)


a = np.ravel(solution['x'])
#in the above vector, the points which have non-zero(or a very low cut off value - we have chosen 1e-5) values are the support vectors.

sv = a > 1e-5 # [true if condition satisfies, else false]
ind = np.arange(len(a))[sv] #getting the indices of the support vectors in the training set
#ex: ind = [1,4,12,59,30] <- 1st,4th,12th etc.. are all support vectors.
a = a[sv] # all support vectors.

sv_x = X[sv]
sv_y = y[sv]

#now we need to compute intercept (b)
#b = ym - sigma xnynK(xn,xm) wehre xn>cutoff and K(xn,xm) is kernalized input
#basically, it is b = acual - predicted (error)
#since we are only using linear kernel, it is Xi^T.Xj (K matrix)

b = 0

for i in range(len(a)):
    ym = sv_y[i]
    xm = sv_x[i]
    b+=ym
    b-=np.sum(a*sv_y*K[sv,ind[i]])
b/= len(a)      

#computing weight vector:
w = np.zeros(n_features)
#w = sigma xn * Xn * yn where [Xn,yn] are the data points of the support vectors and xn is the alpha of the support vector.
#since we are using linear kernel, we need weights.
for i in range(len(a)):
    w += a[i]*sv_x[i]*sv_y[i]





y_predict = np.zeros(len(X_test))
for i in range(len(X_test)):
    #similar to the formula for b just re-arrange it and calculate kernel for test set this time.
    s = 0
    for a_val,xm,ym in zip(a,sv_x,sv_y):
        print a_val 
        print xm
        print kernel(X_test[i],xm)
        s += a_val * ym * kernel(X_test[i],xm)
    y_predict[i] = s

y_predict = np.sign(y_predict + b)

print "accuracy is ",(sum(y_predict==y_test)/len(y_predict))


#right now, it only works on binary classification tasks but one-vs -all technique can be used to implement a multi-classification tasks.

