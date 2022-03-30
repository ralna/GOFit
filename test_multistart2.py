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
    return data[:,1] - fun(p,data[:,0])

print('Norm of residual at minimizer:')
res = eval_res([4,3,2,1])
print(np.sum(res.dot(res)))
print('Residual shape:')
print(res.shape)

# Problem data
m = data.shape[0]
n = 4
p0 = np.array([8, 2.98, 4, 1.02])
pl = np.array([2, 1, 0, -1])
pu = np.array([10, 5, 5, 3])

# Parameters
samples = 100
maxit = 200

# run alternating multistart quadratic regularisation
p, status = multistart(m,n,pl,pu,eval_res,samples=samples,maxit=maxit)

print("Status:")
print(status)
print("p*:")
print(p)
