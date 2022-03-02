/*
 * Finite difference approximations to the Jacobian. See:
 *
 * CITE PRACTICAL OPTIMIZATION
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#ifndef NUMDIFF_H
#define NUMDIFF_H

#include <functional>
#include <Eigen/Core>

/*
 * Forward finite difference approximation to the Jacobian
 *
 * Inputs:
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x -  point at which to estimate the derivative
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 * Outputs:
 *
 *  jac - forward finite difference approximation to the Jacobian at x
 *
 */
void forward_difference_jac(int m, int n, Eigen::VectorXd &x,
                            std::function<void(Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                            Eigen::MatrixXd &jac);

/*
 * Central finite difference approximation to the Jacobian
 *
 * Inputs:
 *
 *  m - number of residuals
 *
 *  n - problem dimension
 *
 *  x -  point at which to estimate the derivative
 *
 *  eval_res - function that evaluates the residual, must have the signature:
 *
 *     void eval_res(Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 * Outputs:
 *
 *  jac - central finite difference approximation to the Jacobian at x
 *
 */
void central_difference_jac(int m, int n, Eigen::VectorXd &x,
                            std::function<void(Eigen::VectorXd&, Eigen::VectorXd&)> eval_res,
                            Eigen::MatrixXd &jac);

/*
 * Compute absolute step size given relative step size and x
 *
 * Returns rel_eps * sign(x)*max(1,abs(x)) for x_i nonzero
 *         rel_eps                         for x_i zero
 *
 * Inputs:
 *
 *  rel_eps - relative step size
 *
 *  x - point at which to estimate the derivative
 *
 * Outputs:
 *
 *  abs_eps - absolute step size
 *
 */
void compute_abs_eps(double rel_eps, Eigen::VectorXd &x, Eigen::VectorXd &abs_eps);

#endif
