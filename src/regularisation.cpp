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
#include <cmath>

#include "regularisation.h"
#include "findiff.h"
#include <Eigen/QR>

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::function;
using std::printf;
using std::min;
using std::max;

// Debug printing
#define VERBOSE false

// function prototypes
void reg(const Control &control, const MatrixXd&, const VectorXd&, double&, VectorXd&);
void reg_update(const Control &control, VectorXd&, double, const VectorXd&, double, const MatrixXd&, const VectorXd&, double&);

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
int regularisation(int m, int n, VectorXd &x,
                   function<void(const VectorXd&, VectorXd&)> eval_res,
                   int maxit /*=200*/, double eps_g /*=1e-4*/, double eps_s /*=1e-8*/){

    // Set control parameters
    Control control;
    control.maxit = maxit;
    control.eps_g = eps_g;
    control.eps_s = eps_s;

    // Forward finite-difference Jacobian
    auto eval_jac = [&eval_res,m,n] (const VectorXd &x, MatrixXd &jac){
        forward_difference_jac(m,n,x,eval_res,jac);
    };

    Inform inform;
    int status = regularisation(control, inform, m, n, x, eval_res, eval_jac);

    return status;
}

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
int regularisation(int m, int n, VectorXd &x,
                   function<void(const VectorXd&, VectorXd&)> eval_res,
                   function<void(const VectorXd&, MatrixXd&)> eval_jac,
                   int maxit /*=200*/, double eps_g /*=1e-4*/, double eps_s /*=1e-8*/){

    // Set control parameters
    Control control;
    control.maxit = maxit;
    control.eps_g = eps_g;
    control.eps_s = eps_s;

    Inform inform;
    int status = regularisation(control, inform, m, n, x, eval_res, eval_jac);

    return status;
}

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
int regularisation(const Control &control, Inform &inform, int m, int n, VectorXd &x,
                   function<void(const VectorXd&, VectorXd&)> eval_res,
                   function<void(const VectorXd&, MatrixXd&)> eval_jac){

    #if VERBOSE
    printf("=+= Quadratic Regularisation Algorithm =+=\n");
    #endif

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

        #if VERBOSE
        printf("\nIteration %d", k);
        printf("\n sigma_k: %.2f", sigma);
        printf("\n s_k: ");
        for(int i = 0; i < n; i++) printf(" %.2f", s(i));
        printf("\n x_k: ");
        for(int i = 0; i < n; i++) printf(" %.2f", x(i));
        printf("\n f(x_k): %.2f", fx);
        printf("\n ||r(x_k)||: %.2f", r.norm());
        printf("\n ||g(x_k)||: %.2f", gradf.norm());
        printf("\n");
        #endif

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
void reg_update(const Control &control, VectorXd &x, double fx, const VectorXd &s, double fxs, const MatrixXd &J, const VectorXd &gradf, double &sigma){

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
void reg(const Control &control, const MatrixXd &J, const VectorXd &gradf, double &sigma, VectorXd &s){

    // Size of Jacobian
    int m = J.rows();
    int n = J.cols();
    int k = min(m,n);

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
    auto qr = perJ.householderQr(); // then R'R = J'J + sigmaI
    auto R = qr.matrixQR().topLeftCorner(k,n).template triangularView<Eigen::UpLoType::Upper>();
    auto RT = qr.matrixQR().topLeftCorner(k,n).template triangularView<Eigen::UpLoType::Upper>().transpose();
    VectorXd t = RT.solve(-gradf); // solve R't = -gradf
    s = R.solve(t);                // solve Rs = t
}
