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
#include <cmath>

#include "regularisation.h"
#include <Eigen/QR>

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::printf;
using std::min;
using std::max;

// Function prototypes
void reg(Control &control, MatrixXd&, VectorXd&, double&, VectorXd&);
void reg_update(Control &control, VectorXd&, double, VectorXd&, double, MatrixXd&, VectorXd&, double&);

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
 *      obj - objective value at minimum
 *      sigma - regularisation parameter value at minimum
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
 *  x - minimal point
 *
 *  return value - 0 (converged) or 1 (iterations exceeded)
 */
int regularisation(Control &control, Inform &inform, int m, int n, VectorXd &x,
                   void (*eval_res)(VectorXd&, VectorXd&),
                   void (*eval_jac)(VectorXd&, MatrixXd&)){

    // Initialisation
    int k = 0;     // iteration counter
    double sigma;  // regularisation parameter
    double fx;     // objective at x
    VectorXd s(n); // step

    // For each iteration
    for(int i = 0; i < control.maxit; i++){

        // Assemble matrices
        VectorXd r(m); // residual
        eval_res(x, r);
        MatrixXd J(m,n); // Jacobian
        eval_jac(x, J);
        fx = 0.5*r.squaredNorm();
        VectorXd gradf = J.transpose()*r;

        // Stop if gradient sufficiently small
        if(gradf.norm() < control.eps_g){
            inform.iter = k;
            inform.obj = fx;
            inform.sigma = sigma;
            return 0;
        }

        // Set initial regularisation parameter
        if(i == 0){
            sigma = gradf.norm()/10;
        }

        // Solve regularisation subproblem to get step s
        reg(control, J, gradf, sigma, s);

        // Stop if step sufficiently small
        if(s.norm() < control.eps_s){
            inform.iter = k;
            inform.obj = fx;
            inform.sigma = sigma;
            return 0;
        }

        // Update regularisation parameter and take step
        VectorXd xs = x+s;
        eval_res(xs, r);
        double fxs = 0.5*r.squaredNorm();
        reg_update(control, x, fx, s, fxs, J, gradf, sigma);

        // Increment iteration counter
        k++;
    }

    // iterations exceeded
    inform.iter = k;
    inform.obj = fx;
    inform.sigma = sigma;
    return 1;
}

 /*
 * Standard regularisation subproblem update
 *
 * Inputs:
 *
 *  control - control parameters (see header file)
 *  x - current iterate
 *  fx - objective at x
 *  s - step
 *  fxs - objective at x+s
 *  J - Jacobian
 *  gradf - objective gradient
 *  sigma - regularisation parameter
 *
 * Outputs:
 *
 *  x - next iterate
 *  sigma - updated regularisation parameter
 *
 */
void reg_update(Control &control, VectorXd &x, double fx, VectorXd &s, double fxs, MatrixXd &J, VectorXd &gradf, double &sigma){

    // Evaluate sufficient decrease
    double Delta_m = -gradf.dot(s) -0.5*(J*s).squaredNorm();
    double rho = (fx - fxs)/(Delta_m - 0.5*sigma*s.squaredNorm());

    // Take step if decrease is sufficient
    if(rho >= control.ETA1){
        x = x + s;
    }

    // Update regularisation parameter
    if(rho < control.ETA1){ // step unsuccessful, more regularisation
        sigma *= control.GAMMA1;
        sigma = min(sigma, control.SIGMA_MAX);
    }else if(rho >= control.ETA2){ // step very successful, less regularisation
        sigma *= control.GAMMA2;
        sigma = max(sigma, control.SIGMA_MIN);
    }
}

/*
 * Solve the quadratic regularisation subproblem using QR
 *
 * See Nocedal & Wright, Chapter 10: Least-Squares Problems
 *
 * Inputs:
 *
 *  control - control parameters (see header file)
 *  J - Jacobian
 *  gradf - objective gradient
 *  sigma - regularisation parameter
 *
 * Outputs:
 *
 *  s - step
 *  sigma - regularisation parameter (maybe modified)
 *
 */
void reg(Control &control, MatrixXd &J, VectorXd &gradf, double &sigma, VectorXd &s){

    // Size of Jacobian
    int m = J.rows();
    int n = J.cols();

    // If J'J singular limit sigma to sigma_lim
    auto qrJ = J.colPivHouseholderQr(); // rank-revealing factorisation
    if(qrJ.rank() < n){
        sigma = max(sigma,control.SIGMA_LIM);
    }

    // Assemble perJ = [J; sqrt(sigma)I]
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd perJ(m+n, n);
    perJ.topRows(m) = J;
    perJ.bottomRows(n) = sqrt(sigma)*I;

    // Solve *perturbed* normal equations (J'J + sigmaI)s = -gradf to find search direction s
    auto qr = perJ.colPivHouseholderQr(); // then R'R = J'J + sigmaI
    auto R = qr.matrixR().topLeftCorner(n,n).template triangularView<Eigen::UpLoType::Upper>();
    auto RT = qr.matrixR().topLeftCorner(n,n).template triangularView<Eigen::UpLoType::Upper>().transpose();
    VectorXd t = RT.solve(-gradf); // solve R't = -gradf
    s = R.solve(t);                // solve Rs = t
}
