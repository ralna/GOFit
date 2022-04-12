/*
 * Python Interface for Alternating multistart adaptive quadratic regularisation. See:
 *
 * O’Flynn, M., Fowkes, J., & Gould, N. (2022).
 * Global optimization of crystal field parameter fitting in Mantid.
 * RAL Technical Reports, RAL-TR-2022-002.
 *
 * Copyright (C) 2022 The Science and Technology Facilities Council (STFC)
 * Author: Jaroslav Fowkes (STFC)
 */

// Includes
#include "alternating.h"

#include <tuple>
#include <stdexcept>
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
using std::invalid_argument;


// Python alternating function that returns x and status
tuple<VectorXd, int> py_alternating(int m, int n, int n_split, const VectorXd &x0,
                                    const VectorXd &xl, const VectorXd &xu,
                                    function<VectorXd(const VectorXd&)> py_eval_res,
                                    int samples, int maxit, double eps_r, double eps_g, double eps_s){

    // Check arguments
    if(m <= 0 || n <= 0 || n_split <= 0 || samples <= 0 || maxit < 0 || eps_r <= 0 || eps_g <= 0 || eps_s <= 0){
        throw invalid_argument("scalar arguments must be strictly positive");
    }else if(x0.size() != n || xl.size() != n || xu.size() != n){
        throw invalid_argument("vector arguments must all be of size n");
    }else if(n_split >= n){
        throw invalid_argument("n_split must be less than n");
    }else if((xl.array() >= xu.array()).any()){
        throw invalid_argument("xu must be strictly greater than xl");
    }else if((xl.array() > x0.array()).any() || (xu.array() < x0.array()).any()){
        throw invalid_argument("x0 must be contained in [xl,xu]");
    }

    // Wrap Python eval_res to C++ signature that we support
    auto eval_res = [&py_eval_res] (const VectorXd &x, VectorXd &res){ res = py_eval_res(x); };

    // Call C++ alternating function
    VectorXd x(n);
    int status = alternating(m, n, n_split, x0, xl, xu, eval_res, x, samples, maxit, eps_r, eps_g, eps_s);

    // Return minimizer and status to Python
    return {x, status};
}

// Python multistart function that returns x and status
tuple<VectorXd, int> py_multistart(int m, int n, const VectorXd &xl, const VectorXd &xu,
                                   function<VectorXd(const VectorXd&)> py_eval_res,
                                   function<MatrixXd(const VectorXd&)> py_eval_jac,
                                   int samples, int maxit, double eps_r, double eps_g, double eps_s,
                                   bool scaling){

    // Check arguments
    if(m <= 0 || n <= 0 || samples <= 0 || maxit < 0 || eps_r <= 0 || eps_g <= 0 || eps_s <= 0){
        throw invalid_argument("scalar arguments must be strictly positive");
    }else if(xl.size() != n || xu.size() != n){
        throw invalid_argument("vector arguments must all be of size n");
    }else if((xl.array() >= xu.array()).any()){
        throw invalid_argument("xu must be strictly greater than xl");
    }

    // Wrap Python eval_res to C++ signature that we support
    auto eval_res = [&py_eval_res] (const VectorXd &x, VectorXd &res){ res = py_eval_res(x); };

    VectorXd x(n);
    int status;
    if(py_eval_jac){ // If Python eval_jac given

        // Wrap Python eval_jac to C++ signature that we support
        auto eval_jac = [&py_eval_jac] (const VectorXd &x, MatrixXd &jac){ jac = py_eval_jac(x); };

        // Call C++ multistart function
        status = multistart(m, n, xl, xu, eval_res, eval_jac, x, samples, maxit, eps_r, eps_g, eps_s, scaling);

    }else{ // Otherwise Python eval_jac not given

        // Call C++ multistart function with finite-difference Jacobian
        status = multistart(m, n, xl, xu, eval_res, x, samples, maxit, eps_r, eps_g, eps_s, scaling);
    }

    // Return minimizer and status to Python
    return {x, status};
}

// Python regularisation function that returns x and status
tuple<VectorXd, int> py_regularisation(int m, int n, VectorXd &x,
                                       function<VectorXd(const VectorXd&)> py_eval_res,
                                       function<MatrixXd(const VectorXd&)> py_eval_jac,
                                       int maxit, double eps_g, double eps_s){

    // Check arguments
    if(m <= 0 || n <= 0 || maxit < 0 || eps_g <= 0 || eps_s <= 0){
        throw invalid_argument("scalar arguments must be strictly positive");
    }else if(x.size() != n){
        throw invalid_argument("vector arguments must all be of size n");
    }

    // Wrap Python eval_res to C++ signature that we support
    auto eval_res = [&py_eval_res] (const VectorXd &x, VectorXd &res){ res = py_eval_res(x); };

    int status;
    if(py_eval_jac){ // If Python eval_jac given

        // Wrap Python eval_jac to C++ signature that we support
        auto eval_jac = [&py_eval_jac] (const VectorXd &x, MatrixXd &jac){ jac = py_eval_jac(x); };

        // Call C++ regularisation function
        status = regularisation(m, n, x, eval_res, eval_jac, maxit, eps_g, eps_s);

    }else{ // Otherwise Python eval_jac not given

        // Call C++ regularisation function with finite-difference Jacobian
        status = regularisation(m, n, x, eval_res, maxit, eps_g, eps_s);
    }

    // Return minimizer and status to Python
    return {x, status};
}


// Main python module
PYBIND11_MODULE(gofit, m) {

    // module docstring
    m.doc() = "GOFit: Global Optimization for Fitting problems";

    // alternating function definition
    m.def("alternating", &py_alternating,

          // position-only arguments
          py::arg("m"),
          py::arg("n"),
          py::arg("n_split"),
          py::arg("x0"),
          py::arg("xl"),
          py::arg("xu"),
          py::arg("res"),
          py::pos_only(),

          // keyword-only arguments with defaults
          py::kw_only(),
          py::arg("samples") = 100,
          py::arg("maxit") = 200,
          py::arg("eps_r") = 1e-5,
          py::arg("eps_g") = 1e-4,
          py::arg("eps_s") = 1e-8,

          // function docstring
          "Alternating multistart adaptive quadratic regularisation"
    );

    // multistart function definition
    m.def("multistart", &py_multistart,

          // position-only arguments
          py::arg("m"),
          py::arg("n"),
          py::arg("xl"),
          py::arg("xu"),
          py::arg("res"),
          py::pos_only(),

          // position and keyword arguments with defaults
          py::arg("jac") = nullptr,

          // keyword-only arguments with defaults
          py::kw_only(),
          py::arg("samples") = 100,
          py::arg("maxit") = 200,
          py::arg("eps_r") = 1e-5,
          py::arg("eps_g") = 1e-4,
          py::arg("eps_s") = 1e-8,
          py::arg("scaling") = true,

          // function docstring
          "Multistart adaptive quadratic regularisation"
    );

    // regularisation function definition
    m.def("regularisation", &py_regularisation,

          // position-only arguments
          py::arg("m"),
          py::arg("n"),
          py::arg("x"),
          py::arg("res"),
          py::pos_only(),

          // position and keyword arguments with defaults
          py::arg("jac") = nullptr,

          // keyword-only arguments with defaults
          py::kw_only(),
          py::arg("maxit") = 200,
          py::arg("eps_g") = 1e-4,
          py::arg("eps_s") = 1e-8,

          // function docstring
          "Adaptive quadratic regularisation"
    );

    // get version number from setup.py
    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
