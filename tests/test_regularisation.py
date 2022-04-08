from gofit import regularisation
import numpy as np

# Load data
data = np.loadtxt('cubic.txt')
print('Data size:')
print(data.shape)

# Simple Least-Squares Test
def fun(p,x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3

def eval_res(p):
    x = data[:,0]
    y = data[:,1]
    return y - fun(p,x)

def eval_jac(p):
    x = data[:,0]
    ones = np.ones(len(x))
    return -1*np.transpose([ones, x, x**2, x**3])

# Problem data
m = data.shape[0]
n = 4
p0 = np.array([8, 2.98, 4, 1.02])

# Parameters
maxit = 200

# run quadratic regularisation
p, status = regularisation(m, n, p0, eval_res, jac=eval_jac, maxit=maxit)

print("Status:")
print(status)
print("p*:")
print(p)

# run quadratic regularisation
p, status = regularisation(m, n, p0, eval_res, maxit=maxit)

print("Status:")
print(status)
print("p*:")
print(p)
