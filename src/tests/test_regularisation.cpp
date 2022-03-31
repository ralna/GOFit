/*
 * Test for the implementation of adaptive quadratic regularisation. See:
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

#include "../regularisation.h"

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
 * Run spec test
 */
int main(){

    printf("=+= Quadratic Regularisation Algorithm =+=\n");

    // Control
    Control control;
    control.eps_g = 1e-4;
    control.eps_s = 1e-8;
    control.maxit = 100;

    // Inform
    Inform inform;

    // Problem data
    int m = 2;
    int n = 2;
    Eigen::VectorXd x {{-1.,1.}};

    // Run on test function
    int status = regularisation(control,inform,m,n,x,eval_res,eval_jac);

    // Output minimum
    if(status == 0){
        printf("Successfully converged!\n");
        printf("Minimum value of %.6f at x = (", inform.obj);
        for(int i = 0; i < n-1; i++) printf("%.8f, ", x(i));
        printf("%.8f)\n", x(n-1));
        printf("found with guaranteed tolerances");
        printf(" in %d iterations.\n", inform.iter);
    }else{
        printf("Maximum number of iterations exceeded!\n");
        printf("Minimum value of %.6f at x = (", inform.obj);
        for(int i = 0; i < n-1; i++) printf("%.8f, ", x(i));
        printf("%.8f)\n", x(n-1));
        printf("found in %d iterations.\n", inform.iter);
    }

    return status;
}
