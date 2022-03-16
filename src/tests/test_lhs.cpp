/*
 * Tests for Latin Hypercube Sampling. See:
 *
 * McKay, M. D., Beckman, R. J., & Conover, W. J. (2000).
 * A comparison of three methods for selecting values of input variables
 * in the analysis of output from a computer code.
 * Technometrics, 42(1), 55-61.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */
#include <iostream>

#include "../lhs.h"

// method aliases
using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(){

    int s = 3; // samples
    int n = 2; // dimension
    MatrixXd lhd(s,n);
    VectorXd a {{-5.,0.}};
    VectorXd b {{10.,15.}};

    cout << "=+= Latin Hypercube Test =+=" << endl;
    cout << "s = " << s << endl;
    cout << "n = " << n << endl;

    lhsdesign(s,n,lhd);

    cout << "Latin Unit Hypercube Design:" << endl;
    cout << lhd << endl;

    lhsdesign(s,n,a,b,lhd);

    cout << "Latin Scaled Hypercube Design:" << endl;
    cout << lhd << endl;

}
