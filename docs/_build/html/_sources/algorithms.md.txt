GOFit Algorithms
================
The GOFit module contains the following three optimization algorithms for parameter fitting nonlinear least-squares problems. The first two are global optimization algorithms and the third is an interface to the local optimization algorithm that is used within the global algorithms. Note that all algorithms expect numpy arrays for vector and matrix inputs.

Multistart Algorithm
--------------------

```{py:function} gofit.multistart(m, n, xl, xu, res [, jac=None, samples=100, maxit=200, eps_r=1e-05, eps_g=1e-04, eps_s=1e-08, scaling=True]) -> (x, status)

Multistart global optimization algorithm. Starts multiple regularisation local optimization algorithms from a given number of randomly generated Latin Hypercube starting points.

:param int m: number of residuals (number of data points)
:param int n: number of parameters (dimension of the problem)
:param numpy.ndarray xl: lower bounds of the parameters to optimize
:param numpy.ndarray xu: upper bounds of the parameters to optimize
:param callable res: function that evaluates the residual, must have the signature `r = res(x)` where `r` is the residual (a *numpy.ndarray* of size *m*) evaluated at `x` (a *numpy.ndarray* of size *n*)
:param jac: optional function that evaluates the Jacobian, must have the signature `J = jac(x)` where `J` is the Jacobian (a *numpy.ndarray* of size *(m,n)*) of the residual evaluated at `x` (a *numpy.ndarray* of size *n*). If not given computes the Jacobian using finite-differences
:type jac: None or callable, optional
:param samples: number of Latin Hypercube starting points for the local solver
:type samples: int, optional
:param maxit: maximum number of iterations for each local solver run
:type maxit: int, optional
:param eps_r: residual stopping tolerance
:type eps_r: float, optional
:param eps_g: norm of gradient stopping tolerance
:type eps_g: float, optional
:param eps_s: norm of step stopping tolerance
:type eps_s: float, optional
:param scaling: whether to scale the optimization parameters (recommended)
:type scaling: bool, optional

:return: optimal parameters, status code
:rtype: (*numpy.ndarray*,*int*)
```

Alternating Algorithm
---------------------

```{py:function} gofit.alternating(m, n, n_split, x0, xl, xu, res [, samples=100, maxit=200, eps_r=1e-05, eps_g=1e-04, eps_s=1e-08]) -> (x, status)

Alternating multistart global optimization algorithm. Assumes the parameters split into `n_split` model parameters and `n-n_split` shape parameters. Then proceeds as follows: 1. fix initial shape params, globally optimize model params; 2. fix model params, locally optimize shape params; 3. locally optimize over model params again; 4. locally optimize over shape params again. Please note that the optimization parameters are scaled by default.

:param int m: number of residuals (number of data points)
:param int n: number of parameters (dimension of the problem)
:param int n_split: parameter split point for alternating optimization (<*n*)
:param numpy.ndarray x0: initial guess for the parameters
:param numpy.ndarray xl: lower bounds of the parameters to optimize
:param numpy.ndarray xu: upper bounds of the parameters to optimize
:param callable res: function that evaluates the residual, must have the signature `r = res(x)` where `r` is the residual (a *numpy.ndarray* of size *m*) evaluated at `x` (a *numpy.ndarray* of size *n*)
:param samples: number of Latin Hypercube starting points for the local solver
:type samples: int, optional
:param maxit: maximum number of iterations for each local solver run
:type maxit: int, optional
:param eps_r: residual stopping tolerance
:type eps_r: float, optional
:param eps_g: norm of gradient stopping tolerance
:type eps_g: float, optional
:param eps_s: norm of step stopping tolerance
:type eps_s: float, optional

:return: optimal parameters, status code
:rtype: (*numpy.ndarray*,*int*)
```

Regularisation Algorithm
------------------------

```{py:function} gofit.regularisation(m, n, x, res [, jac=None, maxit=200, eps_g=1e-04, eps_s=1e-08]) -> (x, status)

Adaptive quadratic regularisation local optimization algorithm. Included for completeness.

:param int m: number of residuals (number of data points)
:param int n: number of parameters (dimension of the problem)
:param numpy.ndarray x0: initial guess for the parameters
:param callable res: function that evaluates the residual, must have the signature `r = res(x)` where `r` is the residual (a *numpy.ndarray* of size *m*) evaluated at `x` (a *numpy.ndarray* of size *n*)
:param jac: optional function that evaluates the Jacobian, must have the signature `J = jac(x)` where `J` is the Jacobian (a *numpy.ndarray* of size *(m,n)*) of the residual evaluated at `x` (a *numpy.ndarray* of size *n*). If not given computes the Jacobian using finite-differences
:type jac: None or callable, optional
:param maxit: maximum number of iterations for each local solver run
:type maxit: int, optional
:param eps_g: norm of gradient stopping tolerance
:type eps_g: float, optional
:param eps_s: norm of step stopping tolerance
:type eps_s: float, optional

:return: optimal parameters, status code
:rtype: (*numpy.ndarray*,*int*)
```
