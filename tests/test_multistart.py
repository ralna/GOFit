from gofit import multistart
import numpy as np

# Levy and Gomez Test Function
def eval_res(x):
    res = np.zeros(3)

    y = 1 + (x - 1)/4
    res[0] = np.sin(np.pi*y[0])
    res[1] = (y[0]-1)*np.sqrt(1+10*np.sin(np.pi*y[1])**2)
    res[2] = y[1]-1

    return res

# Problem data
m = 3
n = 2
xl = -10*np.ones(n)
xu = 10*np.ones(n)

# Parameters
samples = 10
maxit = 100

# run multistart quadratic regularisation
x, status = multistart(m, n, xl, xu, eval_res, samples=samples, maxit=maxit)

print("status:")
print(status)
print("x*:")
print(x)
