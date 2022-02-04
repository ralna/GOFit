/*
 * An implementation of adpative quadratic regularisation. See:
 *
 * Sergeyev, Y. D., & Kvasov, D. E. (2015).
 * A deterministic global optimization using smooth diagonal auxiliary functions.
 * Communications in Nonlinear Science and Numerical Simulation, 21(1-3), 99-111.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#ifndef REGULARISATION_H
#define REGULARISATION_H

/*
 * Regularisation algorithm control parameter structure
 */
struct Control{
    int maxit = 200;     // maximum iterations
    double eps_g = 1e-4; // gradient stopping tolerance
    double eps_s = 1e-8; // step stopping tolerance
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
 *  x - starting point
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(Eigen::VectorXd x, Eigen::VectorXd res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 *  eval_jac - function that evaluates the Jacobian, must have the signature:
 *
 *     void eval_jac(Eigen::VectorXd x, Eigen::MatrixXd jac)
 *
 *   The Jacobian of the residual evaluated at x must be assigned to jac.
 *
 * Outputs:
 *
 *  x - optimal point
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int regularisation(Control &control, VectorXd x,
                   void (*eval_res)(VectorXd, VectorXd),
                   void (*eval_jac)(VectorXd, MatrixXd));

#endif
