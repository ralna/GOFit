/*
 * Multistart adpative quadratic regularisation. See:
 *
 * O’Flynn, M., Fowkes, J., & Gould, N. (2022).
 * Global optimization of crystal field parameter fitting in Mantid.
 * RAL Technical Reports, RAL-TR-2022-002.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */

// Includes
#include "multistart.h"
#include "lhs.h"

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::all;
using std::function;

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
               const VectorXd &xl, const VectorXd &xu, VectorXd &x,
               function<void(const VectorXd&, VectorXd&)> eval_res,
               function<void(const VectorXd&, MatrixXd&)> eval_jac,
               bool disp /*= true*/){

    // Initial variables
    double best_fmin = std::numeric_limits<double>::max();
    double best_sigma;
    int best_status, best_iter, best_run;

    // Perform Latin Hypercube Sampling to choose inital points
    MatrixXd lhd(samples,n);
    lhsdesign(samples,n,xu,xl,lhd);

    // For each Latin Hypercube Sample
    #pragma omp parallel for
    for(int k = 0; k < samples; k++){

        // Get Latin Hypercube Sample
        VectorXd x_k = lhd(k,all);

        // compute residual norm
        VectorXd r(m);
        eval_res(x_k, r);
        double r_norm2 = r.squaredNorm();

        // if residual already small enough no need to optimize
        if(r_norm2 < eps_r){

            // check if we have found a better minimum compared to last stored best minimum
            if(0.5*r_norm2 < best_fmin){
                x = x_k;
                best_fmin = 0.5*r_norm2;
                best_iter = 0;
                best_sigma = std::numeric_limits<double>::quiet_NaN();
                best_status = 0;
                best_run = k;
            }

        }else{ // otherwise need to run a local optimization algorithm

            // run adaptive quadratic regularisation algorithm
            int status = regularisation(control, inform, m, n, x_k, eval_res, eval_jac);

            // check if we have found a better minimum compared to last stored best minimum
            if(inform.obj < best_fmin){
                x = x_k;
                best_fmin = inform.obj;
                best_iter = inform.iter;
                best_sigma = inform.sigma;
                best_status = status;
                best_run = k;
            }

        }

    }

    if(disp){
        if(best_sigma == control.SIGMA_MAX){ // hit SIGMA_MAX
            printf("Warning: More regularisation was required but was unsuccessful. May have failed to converge to global minimum.\n");
        }else if(best_sigma == control.SIGMA_MIN){ // hit SIGMA_MIN
            printf("Warning: May have failed to converge to global minimum.\n");
        }

        printf("Global minimum at %.6f found in %i iterations on run number %i.\n", best_fmin, best_iter, best_run+1);
        printf("Global minimiser:\n");
        for(int i = 0; i < n; i++) printf("%.8f\n", x(i));
    }

    return best_status;
}
