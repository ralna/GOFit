/*
 * An implementation of adaptive quadratic regularisation. See:
 *
 * Gould, N. I., Rees, T., & Scott, J. A. (2019).
 * Convergence and evaluation-complexity analysis of a regularized
 * tensor-Newton method for solving nonlinear least-squares problems.
 * Computational Optimization and Applications, 73(1), 1-35.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#ifndef GOFIT_REGULARISATION_H
#define GOFIT_REGULARISATION_H

#include <functional>
#include <Eigen/Core>

/*
 * Regularisation algorithm control parameter structure
 */
struct Control{

    // General regularisation algorithm controls
    int maxit = 200;     // maximum iterations
    double eps_g = 1e-4; // gradient stopping tolerance
    double eps_s = 1e-8; // step stopping tolerance

    // Regularisation update parameters
    double ETA1 = 0.1;         // decrease rho considered sufficient if above
    double ETA2 = 0.75;        // decrease rho considered very successful if above
    double GAMMA1 = sqrt(2.);  // amount to increase regularisation parameter
    double GAMMA2 = sqrt(0.5); // amount to decrease regularisation parameter
    double SIGMA_MIN = 1e-15;  // minimum regularisation parameter value
    double SIGMA_MAX = 1e20;   // maximum regularisation parameter value

    // Regularisation subproblem solve parameters
    double SIGMA_LIM = 1e-8; // if J'J singular limit sigma to sigma_lim

};

/*
 * Regularisation algorithm inform parameter structure
 */
struct Inform{
    int iter;     // number of iterations perfomed
    double obj;   // objective value at minimum
    double sigma; // regularisation parameter value at minimum
};

/*
 * An implementation of adaptive quadratic regularisation.
 *
 * Inputs:
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x - starting point
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 * Optional Inputs:
 *
 *  maxit - maximum iterations
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
int regularisation(int m, int n, Eigen::VectorXd &x,
                   std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                   int maxit=200, double eps_g=1e-4, double eps_s=1e-8);

/*
 * An implementation of adaptive quadratic regularisation.
 *
 * Inputs:
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x - starting point
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
 * Optional Inputs:
 *
 *  maxit - maximum iterations
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
int regularisation(int m, int n, Eigen::VectorXd &x,
                   std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                   std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jac,
                   int maxit=200, double eps_g=1e-4, double eps_s=1e-8);

/*
 * An implementation of adaptive quadratic regularisation.
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
 *      obj - objective value at minimum
 *      sigma - regularisation paramer value at minimum
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x - starting point
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
 *  x - minimal point
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int regularisation(const Control &control, Inform &inform, int m, int n, Eigen::VectorXd &x,
                   std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                   std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&)> eval_jac);

#endif
