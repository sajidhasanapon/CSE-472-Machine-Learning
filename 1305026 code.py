import numpy as np 
import random
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

np.random.seed(23)

######################################################################################################
n_classes       = 3
n_total_points  = 400

max_pass        = 20
tolerance       = 0.0001

# true parameters for data generation
true_mu      = [[np.random.uniform(0, 50) for i in range(2) ] for i in range(n_classes)]  
true_sigma   = [ [[4.0, 0.0],[0.0, 4.0]] for i in range(n_classes)]
true_W       = [1.0/n_classes for i in range(n_classes)]

# true_W       = [random.random() for i in range(n_classes)]
# s = sum(true_W)
# for x in true_W:
#     x = x / s

# initializing latent factors
mu      = [[np.random.uniform(0, 50) for i in range(2) ] for i in range(n_classes)]
sigma   = [ [[5.0, 0.0],[0.0, 5.0]] for i in range(n_classes)]
W       = [1.0/n_classes for i in range(n_classes)]
p       = [[0 for i in range(n_total_points)] for j in range(n_classes)]
#####################################################################################################


def N_i(x, mu, sigma):

    x = np.matrix(x).T
    mu = np.matrix(mu).T
    sigma = np.matrix(sigma)

    det = np.linalg.det(sigma)
    if det == 0:
        print("COV IS SINGULAR !!!")
        exit()
    inv = np.linalg.inv(sigma)

    mat_mul = -0.5 * np.asscalar((x-mu).T * inv * (x-mu))
    return exp(mat_mul) /  ( (2*pi) * sqrt(det) ) # D = 2


def EM(X):

    ################################################################
    # E-step

    for j in range(n_total_points):
        sum_over_i = 0.0

        for i in range(n_classes):
            N_ij = W[i] * N_i(X[j], mu[i], sigma[i])
            p[i][j] = N_ij
            sum_over_i += N_ij

        for i in range(n_classes):
            p[i][j] =  p[i][j] / sum_over_i
    ################################################################

    ################################################################
    # M-step

    for i in range(n_classes):
        sum_p_ij = 0.0
        for j in range(n_total_points):
            sum_p_ij += p[i][j]

        # calculate mu[i]
        mu[i] = [0.0, 0.0]
        for j in range(n_total_points):
            mu[i][0] += X[j][0] * p[i][j]
            mu[i][1] += X[j][1] * p[i][j]
        mu[i][0] /= sum_p_ij
        mu[i][1] /= sum_p_ij

        # calculate sigma[i]
        sigma[i] = [[0.0, 0.0], [0.0, 0.0]]
        for j in range(n_total_points):
            a = X[j][0] - mu[i][0]
            b = X[j][1] - mu[i][1]

            sigma[i][0][0] = sigma[i][0][0] + p[i][j] * a*a
            sigma[i][0][1] = sigma[i][0][1] + p[i][j] * a*b
            sigma[i][1][0] = sigma[i][1][0] + p[i][j] * a*b
            sigma[i][1][1] = sigma[i][1][1] + p[i][j] * b*b

        sigma[i][0][0] /= sum_p_ij 
        sigma[i][0][1] /= sum_p_ij 
        sigma[i][1][0] /= sum_p_ij 
        sigma[i][1][1] /= sum_p_ij 

        # calculate w
        W[i] = sum_p_ij / n_total_points
    ##################################################################


def log_likelihood():

    L = 0
    for j in range(n_total_points):
        sum_ = 0
        for i in range(n_classes):
            sum_ += W[i] * N_i(X[j], mu[i], sigma[i])
        L += np.log(sum_)

    return L

############################################
##########       DRIVER      ###############
############################################
X = np.empty([0,2])
pp = []

for i in range(n_classes):  
    X_i = np.random.multivariate_normal(true_mu[i], true_sigma[i], int(true_W[i] * n_total_points))
    X_i = X_i.T
    pp.append(X_i)
    plt.plot(X_i[0], X_i[1], 'x')
    X = np.append(X, X_i.T, axis=0)
plt.axis('equal')
plt.show()

np.random.shuffle(X)
X = X.tolist()

n_total_points = len(X)
print("Total points : %d" %(n_total_points))
print("~~~~~~~~~~~~~~~~~~~")

LL_prev = None
for i in range(max_pass):
    print("pass %d" %(i+1))

    EM(X)
    LL_new = log_likelihood()

    if LL_prev != None:
        if abs(LL_new - LL_prev) < tolerance:
            print("CONVERGED!!!")
            break

    LL_prev = LL_new

# plotting the generated points
# and the detected points
for i in pp:
    plt.plot(i[0], i[1], 'x')

mu_t = np.array(mu).T
plt.plot(mu_t[0], mu_t[1], 'o', color="black")

for i in range(n_classes):
    cx = mu[i][0]
    cy = mu[i][1]
    a  = sigma[i][0][0]
    b  = sigma[i][1][1]
    t = np.linspace(0, 2*pi, 20)
    plt.plot(cx+a*np.cos(t), cy+b*np.sin(t))

plt.axis('equal')
plt.show()

# printing results
print("\n\nTrue points : ")
for i in true_mu:
    print("(%.2f, %.2f)" %(i[0], i[1]))
print("########################################")

print("\nDetected points : ")
for i in mu:
    print("(%.2f, %.2f)" %(i[0], i[1]))
print("########################################")


