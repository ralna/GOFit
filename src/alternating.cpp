/*
 * Alternating multistart adaptive quadratic regularisation. See:
 *
 * O’Flynn, M., Fowkes, J., & Gould, N. (2022).
 * Global optimization of crystal field parameter fitting in Mantid.
 * RAL Technical Reports, RAL-TR-2022-002.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */

// Includes
#include "alternating.h"
#include "findiff.h"

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::function;

/*
 * Alternating multistart adaptive quadratic regularisation.
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
int alternating(int m, int n, int n_split, const VectorXd &x0,
                const VectorXd &xl, const VectorXd &xu,
                function<void(const VectorXd&, VectorXd&)> eval_res,
                VectorXd &x, int samples /*=100*/, int maxit /*=200*/,
                double eps_r /*=1e-5*/, double eps_g /*=1e-4*/, double eps_s /*=1e-8*/){

    // Control
    Control control;
    control.eps_g = eps_g;
    control.eps_s = eps_s;
    control.maxit = maxit;

    // Inform
    Inform inform;
    int status;

    // length of parameter blocks for alternating optimization
    int n1 = n_split;   // all model parameters
    int n2 = n-n_split; // all shape parameters

    // scaled alternating optimization outputs
    VectorXd y1(n1), y2(n2);

    // scaled model parameter upper and lower bounds
    VectorXd y1l = VectorXd::Zero(n1);
    VectorXd y1u = VectorXd::Ones(n1);

    // Alternating Optimization
    //     Consists of four stages:
    //     Stages 1 and 2 cover the first optimization pass
    //     Stages 3 and 4 cover the second optimization pass

    // Stage 1: Fix initial shape parameters, optimize model parameters
    auto eval_res1 = [&x0,&xl,&xu,&eval_res,n1,n2] (const VectorXd &y1, VectorXd &res1){
        VectorXd xk(n1+n2);
        xk.head(n1) = xl.head(n1).array() + (xu.head(n1) - xl.head(n1)).array() * y1.array();
        xk.tail(n2) = x0.tail(n2);
        eval_res(xk,res1);
    };

    auto eval_jac1 = [&eval_res1,m,n1] (const VectorXd &y1, MatrixXd &jac1){
        forward_difference_jac(m,n1,y1,eval_res1,jac1);
    };

    // Stage 1: Fix initial shape params, optimize model parameters using multistart
    status = multistart(control,inform,samples,m,n1,eps_r,y1l,y1u,y1,eval_res1,eval_jac1,false);

    // Stage 2: Fix optimized model parameters, optimize shape params from initial guess
    auto eval_res2 = [&y1,&xl,&xu,&eval_res,n1,n2] (const VectorXd &y2, VectorXd &res2){
        VectorXd xk(n1+n2);
        xk.head(n1) = y1;
        xk.tail(n2) = y2;
        xk = xl.array() + (xu - xl).array() * xk.array();
        eval_res(xk,res2);
    };

    auto eval_jac2 = [&eval_res2,m,n2] (const VectorXd &y2, MatrixXd &jac2){
        forward_difference_jac(m,n2,y2,eval_res2,jac2);
    };

    // Stage 2: Fix model parameters, optimize shape parameters
    y2 = x0.tail(n2); // starting point and optimization result
    status = regularisation(control,inform,m,n2,y2,eval_res2,eval_jac2);

    // Stage 3: Fix shape parameters, optimize model parameters
    auto eval_res3 = [&y2,&xl,&xu,&eval_res,n1,n2] (const VectorXd &y3, VectorXd &res3){
        VectorXd xk(n1+n2);
        xk.head(n1) = y3;
        xk.tail(n2) = y2;
        xk = xl.array() + (xu - xl).array() * xk.array();
        eval_res(xk,res3);
    };

    auto eval_jac3 = [&eval_res3,m,n1] (const VectorXd &y3, MatrixXd &jac3){
        forward_difference_jac(m,n1,y3,eval_res3,jac3);
    };

    // Stage 3: Optimize over model parameters again
    VectorXd &y3 = y1; // starting point and optimization result
    status = regularisation(control,inform,m,n1,y3,eval_res3,eval_jac3);

    // Stage 4: Fix model parameters, optimize over shape parameters
    auto eval_res4 = [&y3,&xl,&xu,&eval_res,n1,n2] (const VectorXd &y4, VectorXd &res4){
        VectorXd xk(n1+n2);
        xk.head(n1) = y3;
        xk.tail(n2) = y4;
        xk = xl.array() + (xu - xl).array() * xk.array();
        eval_res(xk,res4);
    };

    auto eval_jac4 = [&eval_res4,m,n2] (const VectorXd &y4, MatrixXd &jac4){
        forward_difference_jac(m,n2,y4,eval_res4,jac4);
    };

    // Stage 4: Optimize over shape parameters again
    VectorXd &y4 = y2; // starting point and optimization result
    status = regularisation(control,inform,m,n2,y4,eval_res4,eval_jac4);

    // Rescale optimized parameters back to original bounds
    x.head(n1) = y3;
    x.tail(n2) = y4;
    x = xl.array() + (xu - xl).array() * x.array();

    return status;
}
