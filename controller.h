/*
 * Controller for multistart adpative quadratic regularisation. See:
 *
 * CITE TECH REPORT
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */
#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "multistart.h"

/*
 * Controller for multistart adpative quadratic regularisation.
 *
 * Inputs:
 *
 *  m - number of residuals (number of data points)
 *
 *  n - number of parameters (dimension of the problem)
 *
 *  split_point - parameter split point for alternating optimization
 *
 *  x0 - initial guess parameters
 *
 *  xl - lower bounds of the parameters to optimize
 *
 *  xu - upper bounds of the parameters to optimize
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 *  samples - number of Latin Hypercube initial points
 *
 *  maxit - maximum iterations
 *
 *  eps_r - residual tolerance
 *
 *  eps_g - gradient stopping tolerance
 *
 *  eps_s - step stopping tolerance
 *
 * Outputs:
 *
 *  x - minimal point
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int controller(int m, int n, int split_point, const VectorXd &x0,
               const VectorXd &xl, const VectorXd &xu,
               function<void(const VectorXd&, VectorXd&)> eval_res,
               int samples=100, int maxit=200,
               double eps_r=1e-5, double eps_g=1e-4, double eps_s=1e-8,
               VectorXd &x);

#endif
