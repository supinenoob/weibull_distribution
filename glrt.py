
import numpy as np



###Median of Weibull###

def median(theta, beta):
   return theta * np.log(2)**(1/beta)


###Delta Method###

def delta1(beta):
    return np.log(2)**(1/beta)


def delta2(beta):
    return (-np.log(np.log(2))*(np.log(2))**(1/beta)) / beta**2


def var(eq1, eq2, var_theta, var_beta, cov):
    return eq1**2 * var_theta + eq2**2 * var_beta + 2*eq1*eq2*cov


###GLRT###

def prod_x(x, beta):
    r = []
    for i in x:
        r.append(i**(beta-1))
    print(r)
    total = 1
    for j in r:
        total = total * j
    print('Total: ', total)
    return total


def beta_x(x, beta):
    r = []
    for i in x:
        r.append(i**beta)
        #print(x)
    total = 0
    for j in r:
        total = total + j
    return total


def sum_x(x):
        r = 0
        for i in x:
            r = r + i
        return r


def glrt(n, sx, bx, px, theta, beta):
    top = theta**beta * np.exp(-(sx/theta) + bx/(theta**beta))
    bottom = beta**n * px
    ratio = top/bottom
    return ratio


###Weibull pdf###

def weibull(count, beta, theta, x):
    w =[]
    for i in range(count):
        w = (beta/theta) * (x[i]/theta)**(beta-1) * np.exp(-(x[i]/theta)**beta)
        #print(w)
    return(w)

def main():

    #Define Variables#
    beta = 1.256472
    theta = 2.035333
    cov = 0.00305318
    var_theta = 0.0167682
    var_beta = 0.00525519

    #Open Data#
    file = open('/Users/TristanTaylor/PycharmProjects/Master_Project/melanoma_data', 'r')
    count = 0
    for lines in file:
        count += 1
    #print(count)
    file.close()
    file = open('/Users/TristanTaylor/PycharmProjects/Master_Project/melanoma_data', 'r')
    x = []
    for lines in file:
        x = np.append(x, float(lines))
    #print(x)

    px = prod_x(x, beta)
    print(px)

    bx = beta_x(x, beta)
    print(bx)

    sx = sum_x(x)
    print(sx)

    #Finding Median#
    med = median(theta, beta)
    print("Median:", med)

    #Finding Standard Error#
    d1 = delta1(beta)
    #print(d1)

    d2 = delta2(beta)
    #print(d2)

    v = var(d1, d2, var_theta, var_beta, cov)
    print("Variance:", v)
    sd = np.sqrt(v)
    print("Standard Error:", sd)

    #GLRT#
    test = glrt(count, sx, bx, px, theta, beta)
    print("GLRT:", test)
    chi = -2*np.log(test)
    print("Chi-Square:", chi)

    #Plotting Graphs for Extra Credit#
    '''w = weibull(count, beta, theta, x)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(w, label='Google 3/19/18 - 3/19/19')
    ax.set(xlabel='Time (Days)', ylabel='25 Days IV')
    ax.legend()
    plt.show()'''

main()