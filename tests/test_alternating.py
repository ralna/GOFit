from gofit import alternating
import numpy as np

# Levy and Gomez combined with Rosenbrock Test Function
def eval_res(x):
    res = np.zeros(5)

    y = 1 + (x - 1)/4
    res[0] = np.sin(np.pi*y[0])
    res[1] = (y[0]-1)*np.sqrt(1+10*np.sin(np.pi*y[1])**2)
    res[2] = y[1]-1
    res[3] = (1-x[2])**2
    res[4] = 100*(x[3]-x[2]**2)**2;

    return res

# Problem data
m = 5
n = 4
n_split = 2
x0 = np.array([5.,5.,-1.,1.])
xl = -10*np.ones(n)
xu = 10*np.ones(n)

# Parameters
samples = 10
maxit = 200

# run alternating multistart quadratic regularisation
x, status = alternating(m, n, n_split, x0, xl, xu, eval_res, samples=samples, maxit=maxit)

print("status:")
print(status)
print("x*:")
print(x)
