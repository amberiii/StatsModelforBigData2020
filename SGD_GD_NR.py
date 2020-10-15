#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg

with open('logistic_digits_train.txt', 'r') as f:
    lines = f.readlines()[1:]
X = []
Y = []
for x in lines:
    data_list = x.strip().split(',')
    data_list = list(map(float, data_list))
    Y.append(data_list[-1])
    X.append(data_list[:-1])
X = np.array(X)
Y = np.array(Y).astype('int')
p = X.shape[1]
C = Y.max() + 1

#N = 100 # number of data
#p = 10 # data dimension
#X = np.random.rand(N, p)
#C = 5 # number of class
#Y = np.random.randint(0, C, N)


def classify(X, beta):
    N = X.shape[0]
    Y_pred = np.zeros([N])
    for i in range(N):
        product = np.exp((beta * X[i][None, :]).sum(-1))
        prob = product / product.sum()
        Y_pred[i] = np.argmax(prob)
    return  Y_pred

def log_likelihood(X, Y, beta):
    N = X.shape[0]
    L = 0
    for i in range(N):
        c = Y[i]
        term1 = (beta[c] * X[i]).sum()
        term2 = 0
        for j in range(C):
            term2 += np.exp((beta[j] * X[i]).sum())
        term2 = np.log(term2)
        L += (term1 - term2)
    return L

def compute_grad(X, Y, beta):
    N = X.shape[0]
    grad = np.zeros([C, p])
    LAMBDA = 1/N # regularization term LAMBDA (beta.T * beta)
    #LAMBDA = 1
    for i in range(N):
        c = Y[i]
        grad_term1 = np.zeros([C, p])
        grad_term2 = np.zeros([C, p])
        grad_term1[c] = X[i]
        term2 = 0
        for j in range(C):
            term2 += np.exp((beta[j] * X[i]).sum())
            grad_term2[j] += np.exp((beta[j] * X[i]).sum()) * X[i]
        grad_term2 = 1 / term2 * grad_term2
        grad -= (grad_term1 - grad_term2) 
        grad += LAMBDA * beta
    return grad

def compute_second_grad(X, Y, beta):
    N = X.shape[0]
    hessian = np.zeros([C*p, C*p])
    #LAMBDA = 1/N # regularization term LAMBDA (beta.T * beta)
    LAMBDA = 0.1
    for i in range(N):
        c = Y[i]
        denumerator = 0 
        for j in range(C):
            denumerator += np.exp((beta[j] * X[i]).sum())
        denumerator = denumerator ** 2
        xixiT = np.matmul(X[i][:,None], X[i][None,:])
        for j in range(C):
            for k in range(C):
                coeff = (np.exp((beta[j]*X[i]).sum())*np.exp((beta[k]*X[i]).sum()))
                hessian[j*p:(j+1)*p, k*p:(k+1)*p] += coeff/denumerator * xixiT

    hessian += LAMBDA * np.diag(np.ones([C*p]))
    return hessian

def newton(X, Y):
    beta = np.zeros([C, p]) # parameters for logistic regression model, each row is beta_j for class_j
    MAX_ITER = 100
    step_size = 1e-4
    Ls = []
    for i in range(MAX_ITER):
        # compute first order grad
        grad1 = compute_grad(X, Y, beta) # grad.shape: [cxp]
        # compute first order grad
        grad2 = compute_second_grad(X, Y, beta) # grad2.shape: [cxp,cxp]
        beta = beta - step_size * np.linalg.solve(grad2, grad1.flatten()).reshape(C, p)
        #beta = beta - step_size * np.linalg.solve(np.eye(C*p), grad1.flatten()).reshape(C, p)
        L = log_likelihood(X, Y, beta)
        Ls.append(L)
        if i % 10 == 0:
            print(f"iteration: {i}/{MAX_ITER}, log likelihood: {L:.5f}")
        if i % 10 == 0:
            test(X, Y, beta)
    return Ls, np.linalg.norm(beta)


def gd(X, Y):
    beta = np.zeros([C, p]) # parameters for logistic regression model, each row is beta_j for class_j
    MAX_ITER = 1000
    step_size = 1e-4
    Ls = []
    for i in range(MAX_ITER):
        grad = compute_grad(X, Y, beta)
        beta = beta - step_size * grad
        L = log_likelihood(X, Y, beta)
        Ls.append(L)
        if i % 10 == 0:
            print(f"iteration: {i}/{MAX_ITER}, log likelihood: {L:.5f}")
        if i % 100 == 0:
            test(X, Y, beta)
    return Ls, beta

def sgd(X, Y):
    beta = np.zeros([C, p]) # parameters for logistic regression model, each row is beta_j for class_j
    MAX_ITER = 1000
    step_size = 1e-4
    batch_size = 4
    Ls = []
    for i in range(MAX_ITER):
        batch_idx = np.random.choice(len(X), batch_size)
        grad = compute_grad(X[batch_idx], Y[batch_idx], beta)
        beta = beta - step_size * grad
        L = log_likelihood(X, Y, beta)
        Ls.append(L)
        if i % 10 == 0:
            print(f"iteration: {i}/{MAX_ITER}, log likelihood: {L:.5f}")
        if i % 100 == 0:
            test(X, Y, beta)
    return Ls, beta
    
def test(X, Y, beta):
    Y_pred = classify(X, beta)
    acc = (Y_pred == Y).mean()
    print(f"acc: {acc:.5f}")

def train(X, Y, method):
    if method == 'sgd':
        Ls, beta = sgd(X, Y)
    elif method == 'gd':
        Ls, beta = gd(X, Y)
    elif method == 'newton':
        Ls, beta = newton(X, Y)

    #test(X, Y, beta)

    plt.plot(Ls)
    plt.savefig('log_likelihood_%s.png' % method)


if __name__ == '__main__':
    train(X, Y, 'newton')
        



# In[ ]:




