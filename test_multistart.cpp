/*
 * Tests for multistart adpative quadratic regularisation. See:
 *
 * CITE TECH REPORT
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */
#include <cmath>

#include "multistart.h"

/*
 * Levy and Gomez Test Function
 */

// Residual
void eval_res(Eigen::VectorXd &x, Eigen::VectorXd &res){
    Eigen::VectorXd y = 1 + (x.array() - 1)/4;
    res(0) = sin(M_PI*y(0));
    res(1) = (y(0)-1)*sqrt(1+10*pow(sin(M_PI*y(1)),2));
    res(2) = y(1)-1;
}
// Jacobian
void eval_jac(Eigen::VectorXd &x, Eigen::MatrixXd &jac){
    Eigen::VectorXd y = 1 + (x.array() - 1)/4;
    double dydx = 0.25;
    jac(0,0) = dydx*M_PI*cos(M_PI*y(0));
    jac(0,1) = 0;
    jac(1,0) = dydx*sqrt(1+10*pow(sin(M_PI*y(1)),2));
    jac(1,1) = dydx*(10*M_PI*sin(M_PI*y(1))*cos(M_PI*y(1)))/sqrt(1+10*pow(sin(M_PI*y(1)),2));
    jac(2,0) = 0;
    jac(2,1) = dydx;
}

/*
 * Run spec test
 */
int main(){

    // Control
    Control control;
    control.eps_g = 1e-4;
    control.eps_s = 1e-8;
    control.maxit = 100;
    double eps_r = 1e-5;

    // Inform
    Inform inform;

    // Problem data
    int m = 3;
    int n = 2;
    int samples = 10;
    Eigen::VectorXd x(n);
    Eigen::VectorXd xl {{-10.,-10.}};
    Eigen::VectorXd xu {{10.,10.}};

    // Run on test function
    int status = multistart(control,inform,samples,m,n,eps_r,xl,xu,x,eval_res,eval_jac);

    return status;
}
