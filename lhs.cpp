/*
 * Latin Hypercube Sampling. See:
 *
 * McKay, M. D., Beckman, R. J., & Conover, W. J. (2000).
 * A comparison of three methods for selecting values of input variables
 * in the analysis of output from a computer code.
 * Technometrics, 42(1), 55-61.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */

// Includes
#include <cmath>
#include <random>

#include "lhs.h"

// method aliases
using Eigen::PermutationMatrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;
using Eigen::seq;
using Eigen::all;
using Eigen::last;
using std::shuffle;

/*
 * Generate Latin Unit Hypercube Design
 *
 * Inputs:
 *
 *  s - number of samples
 *  n - dimension of each sample
 *
 * Outputs:
 *
 *  lhd - latin unit hypercube design
 *
 */
void lhsdesign(int s, int n, MatrixXd &lhd){

    // Initialise random number generator
    std::mt19937 engine(std::random_device{}());

    // Initialise permutation operator
    PermutationMatrix<Dynamic,Dynamic> perm(s);
    perm.setIdentity();

    // Generate partition of [0,1]
    VectorXd part = VectorXd::LinSpaced(s+1,0,1);

    // Compute partition midpoints [0.1666, 0.5, 0.8333]
    VectorXd midpoints = ( part(seq(0,last-1)) + part(seq(1,last)) ) / 2;

    // Randomly permute midpoints per dimension
    for(int i = 0; i <n; i++){
        shuffle(perm.indices().data(), perm.indices().data()+s, engine);
        lhd(all,i) = perm * midpoints;
    }
}
