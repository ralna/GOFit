/*
 * Tests for alternating multistart adaptive quadratic regularisation. See:
 *
 * O’Flynn, M., Fowkes, J., & Gould, N. (2022).
 * Global optimization of crystal field parameter fitting in Mantid.
 * RAL Technical Reports, RAL-TR-2022-002.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC) from the Python code by Megan O'Flynn (STFC)
 */
#include <iostream>

#include "../alternating.h"

// method aliases
using std::cout;
using std::endl;
using Eigen::VectorXd;

/*
 * Levy and Gomez combined with Rosenbrock Test Function
 */

// Residual
void eval_res(const VectorXd &x, VectorXd &res){
    VectorXd y = 1 + (x.array() - 1)/4;
    res(0) = sin(M_PI*y(0));
    res(1) = (y(0)-1)*sqrt(1+10*pow(sin(M_PI*y(1)),2));
    res(2) = y(1)-1;
    res(3) = pow(1-x(2),2);
    res(4) = 100*pow(x(3)-pow(x(2),2),2);
}

/*
 * Test controller
 */
int main(){

    cout << "=+= Alternating Optimization Test =+=" << endl;

    // Problem data
    int m = 5;
    int n = 4;
    int split_point = 2;
    VectorXd x(n);
    VectorXd x0 {{5.,5.,-1.,1.}};
    VectorXd xl {{-10.,-10.,-10.,-10.}};
    VectorXd xu {{10.,10.,10.,10.}};

    // Parameters
    int samples = 10;
    int maxit = 200;

    // Run on test function
    int status = alternating(m, n, split_point, x0, xl, xu, eval_res, x, samples, maxit);

    // Evaluate objective at minimiser
    VectorXd rx(m); // residual
    eval_res(x, rx);
    double fx = 0.5*rx.squaredNorm();

    // Output minimum
    if(status == 0){
        cout << "Successfully converged!" << endl;
        cout << "Minimum value of " << fx << " at x:"<< endl;
        cout << x << endl;
        cout << "found with guaranteed tolerances." << endl;
    }else{
        cout << "Maximum number of iterations exceeded!" << endl;
        cout << "Minimum value of " << fx << " at x:"<< endl;
        cout << x << endl;
        cout << "found without guaranteed tolerances." << endl;
    }

    return status;
}
