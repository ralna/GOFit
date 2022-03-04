/*
 * Multistart adpative quadratic regularisation. See:
 *
 * CITE TECH REPORT
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */
#ifndef MULTISTART_H
#define MULTISTART_H

#include "regularisation.h"

/*
 * Multistart adpative quadratic regularisation
 *
 * Inputs:
 *
 *  control - regularisation control structure with the following members:
 *      maxit - maximum iterations
 *      eps_g - gradient stopping tolerance
 *      eps_s - step stopping tolerance
 *
 *  inform - regularisation inform structure with the following members:
 *      iter - number of iterations perfomed
 *      obj - objective value at minimum
 *      sigma - regularisation parameter value at minimum
 *
 *  samples - number of Latin Hypercube initial points
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  eps_r - residual tolerance
 *
 *  xl - lower bounds for each variable
 *
 *  xu - upper bounds for each variable
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 *  eval_jac - function that evaluates the Jacobian, must have the signature:
 *
 *     void eval_jac(const Eigen::VectorXd &x, Eigen::MatrixXd &jac)
 *
 *   The Jacobian of the residual evaluated at x must be assigned to jac.
 *
 * Outputs:
 *
 *  x - candidate global minimum
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int multistart(const Control &control, Inform &inform, int samples, int m, int n, double eps_r,
               const Eigen::VectorXd &xl, const Eigen::VectorXd &xu, Eigen::VectorXd &x,
               std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
               std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jac,
               bool disp = true);

#endif
