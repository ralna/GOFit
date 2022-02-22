/*
 * An implementation of adpative quadratic regularisation. See:
 *
 * Gould, N. I., Rees, T., & Scott, J. A. (2019).
 * Convergence and evaluation-complexity analysis of a regularized
 * tensor-Newton method for solving nonlinear least-squares problems.
 * Computational Optimization and Applications, 73(1), 1-35.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#ifndef REGULARISATION_H
#define REGULARISATION_H

#include <Eigen/Core>

/*
 * Regularisation algorithm control parameter structure
 */
struct Control{
    int maxit = 200;     // maximum iterations
    double eps_g = 1e-4; // gradient stopping tolerance
    double eps_s = 1e-8; // step stopping tolerance
};

/*
 * Regularisation algorithm inform parameter structure
 */
struct Inform{
    int iter;   // number of iterations perfomed
    double obj; // objective value at optimum
};

/*
 * An implementation of adpative quadratic regularisation.
 *
 * Inputs:
 *
 *  control - control structure with the following members:
 *      maxit - maximum iterations
 *      eps_g - gradient stopping tolerance
 *      eps_s - step stopping tolerance
 *
 *  inform - inform structure with the following members:
 *      iter - number of iterations perfomed
 *      obj - objective value at optimum
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x - starting point
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 *  eval_jac - function that evaluates the Jacobian, must have the signature:
 *
 *     void eval_jac(Eigen::VectorXd &x, Eigen::MatrixXd &jac)
 *
 *   The Jacobian of the residual evaluated at x must be assigned to jac.
 *
 * Outputs:
 *
 *  x - optimal point
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int regularisation(Control &control, Inform &inform, int m, int n, Eigen::VectorXd &x,
                   void (*eval_res)(Eigen::VectorXd&, Eigen::VectorXd&),
                   void (*eval_jac)(Eigen::VectorXd&, Eigen::MatrixXd&));

#endif
