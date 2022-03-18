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
#ifndef GOFIT_LHS_H
#define GOFIT_LHS_H

#include <Eigen/Core>

/*
 * Generate Latin Hypercube Design on [0,1]^n
 *
 * Inputs:
 *
 *  s - number of samples
 *  n - dimension of each sample
 *
 * Outputs:
 *
 *  lhd - latin hypercube design on [0,1]^n
 *
 */
void lhsdesign(int s, int n, Eigen::MatrixXd &lhd);

/*
 * Generate Latin Hypercube Design on [a,b]^n
 *
 * Inputs:
 *
 *  s - number of samples
 *  n - dimension of each sample
 *  a - domain lower bounds
 *  b - domain upper bounds
 *
 * Outputs:
 *
 *  lhd - latin hypercube design on [a,b]^n
 *
 */
void lhsdesign(int s, int n, const Eigen::VectorXd &a, const Eigen::VectorXd &b,
               Eigen::MatrixXd &lhd);

#endif
