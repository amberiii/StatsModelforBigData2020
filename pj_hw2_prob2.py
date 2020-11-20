#############################################
#    SDS385 Homework Assignment 2.2 - PJ    #
#############################################

# Load packages and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('noisy.txt', 'r') as rfile:
    data = np.array(list(map(float, rfile.read().split())))

################################################
# 2.c) Implement Subgradient descent algorithm.
################################################

def get_sub_grad(x, z, lamda):
    n = z.shape[0]
    D = np.zeros((n-1,n))
    for i in range(n-1):
        D[i,i] = -1
        D[i,i+1] = 1
    y = np.dot(D,x)
    sgn = np.zeros_like(y)
    for i in range(n-1):
        if y[i] > 0:
            sgn[i] = 1
        elif y[i] < 0:
            sgn[i] = -1
        else:
            sgn[i] = 0
    sub_grad = x - z + lamda * np.dot(D.T, sgn)
    return sub_grad

def sub_GD(z, lamda, lr, tol = 0.00001):
    x = np.zeros_like(z)
    itr = 0
    epsilon = 1
    eps_list = []
    iters_list = []
    while itr < 5000 and epsilon > tol:
        grad = get_sub_grad(x, z, lamda)
        x_new = x - lr * grad
        epsilon = np.linalg.norm(x_new - x)
        x = x_new
        itr = itr + 1
        if epsilon <= tol:
            print("it converges")
        eps_list.append(epsilon)
        iters_list.append(itr)
    return x, eps_list, iters_list

# Figure 1: Sub_GD of learning rate at a set value of 0.001
'''
x, eps, itrs = sub_GD(data, lamda = 0.5, lr = 1e-3)
plt.plot(itrs,eps,label = "step size/lr of 1e-3")
plt.title("lambda = 0.5")
plt.xlabel('Number of iterations')
plt.ylabel('Convergence')
plt.yscale('log')
plt.legend()
plt.show()
'''

# Figure 2: Sub_GD all chosen learning rates together
'''
lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
for i in range(len(lr_list)):
    x, eps, itrs = sub_GD(data, lamda = 0.5, lr = lr_list[i])
    print("plotting lr at index %i" %i)
    plt.plot(itrs,eps,label = f'step size/lr of {lr_list[i]}')
plt.title("Diff learning rates when lambda = 0.5 in subgradient algo")
plt.xlabel('Number of iterations')
plt.ylabel('Convergence')
plt.yscale('log')
plt.legend()
plt.show()
'''

###############################################
#2.e) Implement the Proximal Gradient Algorithm
###############################################
def get_dual_prox(z, lr, lamda, tol=0.00001):
    n = len(z)
    D = np.zeros((n-1,n))
    for i in range(n-1):
        D[i,i] = -1
        D[i,i+1] = 1
    u_prev = np.zeros((n-1,))
    itr = 0
    epsilon = 1
    eps_list = []
    iters_list = []
    while itr < 5000:
        itr = itr + 1
        #update u
        u_new_prox = u_prev - lr * (np.dot(D,np.dot(D.T,u_prev)) - np.dot(D,z)) 
        for i in range(n-1):
            if u_new_prox[i] >= lamda:
                u_new_prox[i] = lamda
            elif u_new_prox[i] < -lamda:
                u_new_prox[i] = -lamda
        #calculate the loss
        #import ipdb;ipdb.set_trace()
        epsilon = np.linalg.norm(u_new_prox - u_prev)
        u_prev = u_new_prox
        #when to converge
        if epsilon <= tol:
            print("Yeah it converges.")
        eps_list.append(epsilon)
        iters_list.append(itr)
    x = z - np.dot(D.T,u_prev)
    return x, eps_list, iters_list

# Figure 3: Dual proximal at lr=0.001
'''
x, eps, itrs = get_dual_prox(data, lr = 0.001, lamda = 0.5)
plt.plot(itrs,eps,label = "step size/lr of 1e-3")
plt.title("Dual Proximal Gradient Algo when lambda = 0.5, y-axis is not scaled")
plt.xlabel('Number of iterations')
plt.ylabel('Convergence')
#plt.yscale('log')
plt.legend()
plt.show()

'''

# Figure 4: Dual Proximal in different learning rates
'''
lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
for i in range(len(lr_list)):
    x, eps, itrs = get_dual_prox(data, lamda = 0.5, lr = lr_list[i])
    print("plotting lr at index %i" %i)
    plt.plot(itrs,eps,label = f'Dual Prox: step size/lr of {lr_list[i]}')
plt.title("Dual Proximal Gradient Algo when lambda = 0.5, y-axis is not scaled")
'''

# Figure 5: Compare subgradient algo with dual proximal algorithm
'''
x, eps, itrs = sub_GD(data, lamda = 0.5, lr = 0.001)
plt.plot(itrs,eps,'--',label = "sub_GD: step size/lr of 1e-3" )

x, eps, itrs = get_dual_prox(data, lr = 0.001, lamda = 0.5)
plt.plot(itrs,eps,label = "Dual Prox: step size/lr of 1e-3")

plt.title("Comparison of the two alogs when lambda=0.5 and lr=0.001")
plt.xlabel('Number of iterations')
plt.ylabel('Convergence')
plt.yscale('log')
plt.legend()
plt.show()
'''
#################################### 
##2.f) Dual Prox, diffrent lambda
####################################


# Figure 6: Denoised data for Dual Prox when lamda = 0.5
'''
x, eps, itrs = get_dual_prox(data, lamda = 0.5, lr = 0.001)
plt.plot(data,label='Original')
plt.plot(x,label='After Dual Prox')
plt.title(" Denoised and Original curve for Dual_Prox at lambda = 0.5 and lr = 0.001")
plt.legend()
plt.show()
'''

# Figure 7: Data and denoised data for Dual Prox with different lamdas

fig, axs = plt.subplots(5, 1, figsize=(5, 6), tight_layout=True)
lamda_list = [0.01, 0.1, 0.05, 0.75, 1]
for ax, l in zip(axs, lamda_list):
    x, eps, itrs = get_dual_prox(data, lamda = l, lr = 0.001)
    ax.plot(data)
    ax.plot(x)
    ax.set_title(f'lamda = {l}')
    
plt.show()

print('Done!')