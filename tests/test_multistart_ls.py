from gofit import multistart
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
n_split = 2
p0 = np.array([8, 2.98, 4, 1.02])
pl = np.array([2, 1, 0, -1])
pu = np.array([10, 5, 5, 3])

# Parameters
samples = 10
maxit = 200

# run multistart quadratic regularisation (no scaling)
p, status = multistart(m, n, pl, pu, eval_res, jac=eval_jac, samples=samples, maxit=maxit, scaling=False)

print("Status:")
print(status)
print("p*:")
print(p)

# run multistart quadratic regularisation (no scaling)
p, status = multistart(m, n, pl, pu, eval_res, samples=samples, maxit=maxit, scaling=False)

print("Status:")
print(status)
print("p*:")
print(p)

# run multistart quadratic regularisation (scaling)
p, status = multistart(m, n, pl, pu, eval_res, jac=eval_jac, samples=samples, maxit=maxit)

print("Status:")
print(status)
print("p*:")
print(p)

# run multistart quadratic regularisation (scaling)
p, status = multistart(m, n, pl, pu, eval_res, samples=samples, maxit=maxit)

print("Status:")
print(status)
print("p*:")
print(p)
