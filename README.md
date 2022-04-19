
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://img.shields.io/github/workflow/status/ralna/gofit/Tests)](https://github.com/ralna/gofit/actions/workflows/test.yml)
[![PyPI Version](https://img.shields.io/pypi/v/gofit.svg)](https://pypi.python.org/pypi/gofit)

GOFit: Global Optimization for Fitting problems
===============================================

GOFit is a package of C++ algorithms with python interfaces designed for the global optimization of parameters in curve fitting, i.e. for nonlinear least-squares problems arising from curve fitting. GOFit was developed with scientific curve fitting problems in mind but is also applicable to general curve fitting problems provided they can be formulated as nonlinear least-squares problems.

Full details on how to use GOFit are available in the [documentation](https://ralna.github.io/gofit/).

Requirements
------------
GOFit provides pre-built [Python wheels](https://realpython.com/python-wheels/) for most common platforms with all dependencies included. However if a wheel does not exist for your platform (or if building from source), GOFit requires the following software to be installed:

* PyBind11 2.9.1 or above (<https://pybind11.readthedocs.io/>)
* Eigen 3.4 or above (<https://eigen.tuxfamily.org/>)
* CMake 3.18 or above (<https://cmake.org/>)

Installing GOFit
----------------
For easy installation, use [pip](http://www.pip-installer.org/):

```bash
$ pip install gofit
```

Note that if an older install of GOFit is present on your system you can use:

```bash
$ pip install --upgrade gofit
```

to upgrade GOFit to the latest version.

Installing GOFit from source
----------------------------
Alternatively, you can download the source code from [Github](https://github.com/ralna/gofit) and unpack as follows:

```bash
$ git clone https://github.com/ralna/gofit
$ cd gofit
```

GOFit can then be compiled and installed using:

```bash
$ pip install .
```

**Please Note:** *don't forget to install the required dependencies (see above).*

To upgrade GOFit to the latest version, navigate to the top-level directory (i.e. the one containing `setup.py`) and re-run the installation using `pip`, as above:

```bash
$ git pull
$ pip install .
```

Testing
-------
The [documentation](https://ralna.github.io/gofit/) provides some simple examples of how to run GOFit.

Uninstallation
--------------
You can uninstall GOFit as follows:

```bash
$ pip uninstall gofit
```

Bugs
----
Please report any bugs using GitHub's issue tracker.

License
-------
This software is released under the New BSD license.
