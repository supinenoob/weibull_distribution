
import numpy as np
from math import *
import math
from numpy.linalg import inv


###Finding the M.L.E.###
def theta_coefficient(beta, x, n):
    return (math.fsum(x**beta)/n)**(1/beta)

def beta_coefficient(beta, x, n):
    return (math.fsum(x**beta*np.log(x)))/(math.fsum(x**beta))-(1/beta)-(math.fsum(np.log(x))/n)

def beta_prime(beta, x):
    db = math.fsum(x**beta*(np.log(x))**2)/math.fsum(x**beta)-(math.fsum(x**beta*np.log(x)))**2/(math.fsum(x**beta))**2+(1/beta**2)
    return db

def newton(x, n, beta):
    percision = 0.000001
    fx = beta_coefficient(beta, x, n)
    dfx = beta_prime(beta, x)
    beta_mle = beta - fx/dfx
    print(beta_mle)
    if abs(beta-beta_mle) > percision:
        #b = b - beta_coefficient(b, x, n)/beta_prime(b, x)
        beta_mle, theta_mle = newton(x, n, beta_mle)
    else:
        print('MLE for Beta is: %f' % (beta_mle))
        theta_mle = theta_coefficient(beta_mle, x, n)
        print('MLE for Theta is: %f' % (theta_mle))

    return beta_mle, theta_mle

###Deriving the Hessian Matrix###
def fisher_scale(beta, theta, x, n):
    return ((n*beta)/theta**2)-(beta/theta**2)*np.sum((x/theta)**beta)-(beta**2/theta)*np.sum((x/theta)**(beta-1)*(x/theta**2))

def fisher_corr1(beta, theta, x, n):
    return (-n/theta)+(1/theta)*np.sum((x/theta)**beta)+(beta/theta)*np.sum((x/theta)**beta*np.log(x/theta))

def fisher_shape(beta, theta, x, n):
    return (-n/beta**2)-np.sum((x/theta)**beta*np.log(x/theta)**2)

def fisher_corr2(beta, theta, x, n):
    return (-n/theta)+(1/theta)*np.sum((x/theta)**beta)+(beta/theta)*np.sum((x/theta)**beta*np.log(x/theta))

def hessian(dt2, dtb, db2, dbt):
    H = np.array([[dt2, dtb],[dbt,db2]])
    return H

##Open Melanoma Dataset##
def main():
    file = open('/Users/TristanTaylor/PycharmProjects/Master_Project/melanoma_data','r')
    count = 0
    for lines in file:
        count += 1
    #print(count)
    file.close()
    file = open('/Users/TristanTaylor/PycharmProjects/Master_Project/melanoma_data','r')
    x = []
    for lines in file:
        x = np.append(x, float(lines))
    #print(x)

    beta_mle, theta_mle = newton(x,count, 1)
    #print(beta_mle)
    #print(theta_mle)
    fisher_theta = fisher_scale(beta_mle, theta_mle, x, count)
    #print(fisher_theta)
    theta_beta = fisher_corr1(beta_mle, theta_mle, x, count)
    #print(theta_beta)
    fisher_beta = fisher_shape(beta_mle, theta_mle, x, count)
    #print(fisher_beta)
    beta_theta = fisher_corr2(beta_mle, theta_mle, x, count)
    #print(beta_theta)

    H = hessian(fisher_theta, theta_beta, fisher_beta, beta_theta)
    print(H)

    IH = -inv(H)
    print(IH)

    sd_theta = sqrt(abs(IH.item(0)))
    print("SD for Theta:" , sd_theta)
    sd_beta = sqrt(abs(IH.item(3)))
    print("SD for Beta:" , sd_beta)
    #print(beta_mle)
    #print('MLE for Beta is: %f' % (beta_mle))

    #theta_mle = theta_coefficient(newton(x, count, 1), x, count)
    #print(theta_mle)
    #print('MLE for Theta is: %f' % (theta_mle))



main()