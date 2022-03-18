/*
 * Finite difference approximations to the Jacobian. See:
 *
 * Gill, P. E., Murray, W., & Wright, M. H. (2019).
 * Practical Optimization. Chapter 8: Practicalities.
 * Society for Industrial and Applied Mathematics.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#include <cmath>

// Includes
#include "findiff.h"

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::all;
using std::function;

/*
 * Forward finite difference approximation to the Jacobian
 *
 * Cost: n+1 residual evaluations, Accuracy: epsilon^1/2
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
 *     void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 * Outputs:
 *
 *  jac - forward finite difference approximation to the Jacobian at x
 *
 */
void forward_difference_jac(int m, int n, const VectorXd &x,
                            function<void(const VectorXd&, VectorXd&)> eval_res,
                            MatrixXd &jac){

    // optimal relative step size is sqrt of machine precision
    const double REL_EPS = sqrt(std::numeric_limits<double>::epsilon());

    // calculate absolute step size
    VectorXd eps(n);
    compute_abs_eps(REL_EPS,x,eps);

    // evaluate residual at x
    VectorXd rx(m);
    eval_res(x, rx);

    // calculate forward finite difference for each coordinate
    VectorXd rx_epsi(m);
    for(int i = 0; i < n; i++){

        // evaluate residual at x + eps(i)*e_i
        VectorXd x_epsi = x + eps(i) * VectorXd::Unit(n,i);
        eval_res(x_epsi, rx_epsi);

        // forward finite difference formula
        jac(all,i) = (rx_epsi - rx) / eps(i);
    }

}

/*
 * Central finite difference approximation to the Jacobian
 *
 * Cost: 2n+1 residual evaluations, Accuracy: epsilon^2/3
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
 *     void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res)
 *
 *   The value of the residual evaluated at x must be assigned to res.
 *
 * Outputs:
 *
 *  jac - central finite difference approximation to the Jacobian at x
 *
 */
void central_difference_jac(int m, int n, const VectorXd &x,
                            function<void(const VectorXd&, VectorXd&)> eval_res,
                            MatrixXd &jac){

    // optimal relative step size is cbrt of machine precision
    const double REL_EPS = cbrt(std::numeric_limits<double>::epsilon());

    // calculate absolute step size
    VectorXd eps(n);
    compute_abs_eps(REL_EPS,x,eps);

    // calculate forward finite difference for each coordinate
    VectorXd rx_epsi(m);
    VectorXd rx_mepsi(m);
    for(int i = 0; i < n; i++){

        // ith unit vector
        VectorXd ei = VectorXd::Unit(n,i);

        // evaluate residual at x + eps(i)*e_i
        VectorXd x_epsi = x + eps(i) * ei;
        eval_res(x_epsi, rx_epsi);

        // evaluate residual at x - eps(i)*e_i
        VectorXd x_mepsi = x - eps(i) * ei;
        eval_res(x_mepsi, rx_mepsi);

        // central finite difference formula
        jac(all,i) = (rx_epsi - rx_mepsi) / (2*eps(i));
    }

}

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
void compute_abs_eps(double rel_eps, const VectorXd &x, VectorXd &abs_eps){

    // compute sign of x
    VectorXd sign_x = x.cwiseSign();

    // need sign to be 1 when x_i is zero
    sign_x = sign_x.cwiseEqual(0).select(1.0,sign_x);

    // compute max(1,abs(x))
    VectorXd abs_x = x.cwiseAbs().cwiseMax(1.0);

    // put it all together
    abs_eps = rel_eps * sign_x.array() * abs_x.array();

}
