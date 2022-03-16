/*
 * Python Interface for Alternating multistart adpative quadratic regularisation. See:
 *
 * CITE TECH REPORT
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */

// Includes
#include "alternating.h"

#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// macros (yes we need both)
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// method aliases
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::function;
using std::tuple;

// Wrap alternating function to return x and status in python
tuple<VectorXd, int> wrap_alternating(int m, int n, int n_split, const VectorXd &x0,
                                      const VectorXd &xl, const VectorXd &xu,
                                      function<VectorXd(const VectorXd&, int)> eval_res,
                                      int samples, int maxit, double eps_r, double eps_g, double eps_s){

    // Wrap python eval_res to signature that we support
    auto wrap_eval_res = [&eval_res,m] (const VectorXd &x, VectorXd &res){ res = eval_res(x,m); };

    VectorXd x(n);
    int status = alternating(m, n, n_split, x0, xl, xu, wrap_eval_res, x, samples, maxit, eps_r, eps_g, eps_s);

    return {x, status};
}

// Main python module
PYBIND11_MODULE(globfit, m) {

    // module docstring
    m.doc() = "GlobFit: Global optimization for Fitting problems";

    // alternating function definition
    m.def("alternating", &wrap_alternating,

          // position-only arguments
          py::arg("m"), py::pos_only(),
          py::arg("n"), py::pos_only(),
          py::arg("n_split"), py::pos_only(),
          py::arg("x0"), py::pos_only(),
          py::arg("xl"), py::pos_only(),
          py::arg("xu"), py::pos_only(),
          py::arg("eval_res"), py::pos_only(),

          // keyword-only arguments
          py::kw_only(),
          py::arg("samples") = 100,
          py::arg("maxit") = 200,
          py::arg("eps_r") = 1e-5,
          py::arg("eps_g") = 1e-4,
          py::arg("eps_s") = 1e-8,

          // function docstring
          "Alternating multistart adpative quadratic regularisation");

    // get version number from setup.py
    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
