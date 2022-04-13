Example Usage
=============
The following code presents a simple use of GOFit to globally minimize, using the [Multistart Algorithm](algorithms.md#multistart-algorithm), the Levy and Gomez test function in 2D.

```python
from gofit import multistart
import numpy as np

# Levy and Gomez test function
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

# Run GOFit multistart algorithm
x, status = multistart(m, n, xl, xu, eval_res, samples=samples, maxit=maxit)

print("status:")
print(status)
print("x*:")
print(x)
```

For this problem, GOFit’s [Multistart Algorithm](algorithms.md#multistart-algorithm) finds the global minimum at `x=[1,1]` quickly:

```text
status:
0
x*:
[1.         1.00000231]
```

More usage examples can be found in the `tests` sub-directory. See [GOFit Algorithms](algorithms.md#gofit-algorithms) for details of the available algorithms.
