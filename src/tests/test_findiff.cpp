/*
 * Tests for finite difference approximations to the Jacobian. See:
 *
 * Gill, P. E., Murray, W., & Wright, M. H. (2019).
 * Practical Optimization. Chapter 8: Practicalities.
 * Society for Industrial and Applied Mathematics.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#include <iostream>

// Includes
#include "../findiff.h"

// method aliases
using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Rosenbrock Test Function
 */

// Residual
void eval_res(const Eigen::VectorXd &x, Eigen::VectorXd &res){
    res(0) = pow(1-x(0),2);
    res(1) = 100*pow(x(1)-pow(x(0),2),2);
}
// Jacobian
void eval_jac(const Eigen::VectorXd &x, Eigen::MatrixXd &jac){
    jac(0,0) = 2*(1-x(0))*(-1);
    jac(0,1) = 0;
    jac(1,0) = 200*(x(1)-pow(x(0),2))*(-2*x(0));
    jac(1,1) = 200*(x(1)-pow(x(0),2));
}

/*
 * Run tests
 */
int main(){

    cout << "=+= Relative to Absolute Step Size Test =+=" << endl;

    double rel_eps = 2.;
    VectorXd eps(6);
    VectorXd xt {{-0.3,1.4,-3.5,0.,0.23,-0.}};
    compute_abs_eps(rel_eps,xt,eps);

    cout << "rel_eps = " << rel_eps << endl;
    cout << "x:" << endl;
    cout << xt << endl;
    cout << "abs_eps:" << endl;
    cout << eps << endl;
    cout << endl;

    int m = 2; // samples
    int n = 2; // dimension
    MatrixXd jac_approx(m,n);
    MatrixXd jac_exact(m,n);
    VectorXd x0 {{0.,0.}};
    VectorXd x {{-1.,1.}};

    cout << "=+= Forward Finite Difference Test =+=" << endl;

    cout << "x:" << endl;
    cout << x0 << endl;
    forward_difference_jac(m,n,x0,eval_res,jac_approx);
    cout << "jac_approx:" << endl;
    cout << jac_approx << endl;
    eval_jac(x0,jac_exact);
    cout << "jac_exact:" << endl;
    cout << jac_exact << endl;
    cout << endl;

    cout << "x:" << endl;
    cout << x << endl;
    forward_difference_jac(m,n,x,eval_res,jac_approx);
    cout << "jac_approx:" << endl;
    cout << jac_approx << endl;
    eval_jac(x,jac_exact);
    cout << "jac_exact:" << endl;
    cout << jac_exact << endl;
    cout << endl;

    cout << "=+= Central Finite Difference Test =+=" << endl;

    cout << "x:" << endl;
    cout << x0 << endl;
    central_difference_jac(m,n,x0,eval_res,jac_approx);
    cout << "jac_approx:" << endl;
    cout << jac_approx << endl;
    eval_jac(x0,jac_exact);
    cout << "jac_exact:" << endl;
    cout << jac_exact << endl;
    cout << endl;

    cout << "x:" << endl;
    cout << x << endl;
    central_difference_jac(m,n,x,eval_res,jac_approx);
    cout << "jac_approx:" << endl;
    cout << jac_approx << endl;
    eval_jac(x,jac_exact);
    cout << "jac_exact:" << endl;
    cout << jac_exact << endl;
    cout << endl;

}