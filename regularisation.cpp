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
#include <cmath>
#include <Eigen/Core>
#include <Eigen/QR>

#include "regularisation.h"

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::printf;
using std::min;
using std::max;

// Debug printing
#define VERBOSE false


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
                   void (*eval_jac)(VectorXd, MatrixXd)){

    printf("=+= Quadratic Regularisation Algorithm =+=\n");

    // Initialisation
    int k = 0;    // iteration counter
    double sigma; // regularisation parameter, objective at x and x+s
    VectorXd s;   // step

    // For each iteration
    for(int i = 0; i < control.maxit; i++){

        // Assemble matrices
        VectorXd r; // residual
        eval_res(x, r);
        MatrixXd J; // Jacobian
        eval_jac(x, J);
        double fx = 0.5*r.squaredNorm();
        VectorXd gradf = J.transpose()*r;

        // Stop if gradient sufficiently small
        if(gradf.norm() < control.eps_g){
            return 0;
        }

        // Set initial regularisation parameter
        if(i == 0){
            sigma = gradf.norm()/10;
        }

        // Solve regularisation subproblem to get step s
        reg(J, gradf, sigma, s);

        // Stop if step sufficiently small
        if(s.norm() < control.eps_s){
           return 0;
        }

        // Update regularisation parameter and take step
        eval_res(x+s, r);
        double fxs = 0.5*r.squaredNorm();
        reg_update(x, fx, s, fxs, J, gradf, sigma);

        // Increment iteration counter
        k++;
    }

    // iterations exceeded
    return 1;
}

 /*
 * Standard regularisation subproblem update
 *
 * Inputs:
 *
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
void reg_update(VectorXd x, double fx, VectorXd s, double fxs, MatrixXd J, VectorXd gradf, double &sigma){

    // Parameters
    const double ETA1 = 0.1;
    const double ETA2 = 0.75;
    const double GAMMA1 = sqrt(2.);
    const double GAMMA2 = sqrt(0.5);
    const double SIGMA_MIN = 1e-15;
    const double SIGMA_MAX = 1e20; // larger for CrystalField problems

    // Evaluate sufficient decrease
    double Delta_m = -gradf.dot(s) -0.5*(J*s).squaredNorm();
    double rho = (fx - fxs)/(Delta_m - 0.5*sigma*s.squaredNorm());

    // Take step if decrease is sufficient
    if(rho >= ETA1){
        x = x + s;
    }

    // Update regularisation parameter
    if(rho < ETA1){
        sigma *= GAMMA1;
        sigma = min(sigma, SIGMA_MAX);
    }else if(rho >= ETA2){
        sigma *= GAMMA2;
        sigma = max(sigma, SIGMA_MIN);
    }
}

/*
 * Solve the quadratic regularisation subproblem using QR
 *
 * See Nocedal & Wright, Chapter 10: Least-Squares Problems
 *
 * Inputs:
 *
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
void reg(MatrixXd J, VectorXd gradf, double& sigma, VectorXd s){

    // Parameters
    const double SIGMA_MIN = 1e-8;

    // Size of Jacobian
    int m = J.rows();
    int n = J.cols();

    // If J'J singular limit sigma to sigma_min
    auto qr = J.colPivHouseholderQr(); // rank-revealing factorisation
    if(qr.rank() < n){
        sigma = max(sigma,SIGMA_MIN);
    }

    // Assemble perJ = [J; sqrt(sigma)I]
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd perJ(m+n, n+n);
    perJ.topRows(J.rows());
    perJ.bottomRows((sqrt(sigma)*I).rows());

    // Solve *perturbed* normal equations (J'J + sigmaI)s = -gradf to find search direction s
    auto qr = perJ.colPivHouseholderQr(); // then R'R = J'J + sigmaI
    auto R = qr.matrixR().topLeftCorner(n,n).template triangularView<Eigen::UpLoType::Upper>();
    auto t = R.transpose().solve(-gradf); // solve R't = -gradf
    s = R.solve(t);                       // solve Rs = t
}
