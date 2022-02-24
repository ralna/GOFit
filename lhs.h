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
#ifndef LHS_H
#define LHS_H

#include <Eigen/Core>

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
void lhsdesign(int s, int n, Eigen::MatrixXd &lhd);

#endif
