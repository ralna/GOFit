/*
 * Alternating multistart adpative quadratic regularisation. See:
 *
 * CITE TECH REPORT
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */
#ifndef GLOBFIT_ALTERNATING_H
#define GLOBFIT_ALTERNATING_H

#include "multistart.h"

/*
 * Alternating multistart adpative quadratic regularisation.
 *
 * Inputs:
 *
 *  m - number of residuals (number of data points)
 *
 *  n - number of parameters (dimension of the problem)
 *
 *  n_split - parameter split point for alternating optimization (<n)
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
 * Optional Inputs:
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
int alternating(int m, int n, int n_split, const Eigen::VectorXd &x0,
                const Eigen::VectorXd &xl, const Eigen::VectorXd &xu,
                std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                Eigen::VectorXd &x, int samples=100, int maxit=200,
                double eps_r=1e-5, double eps_g=1e-4, double eps_s=1e-8);

#endif
