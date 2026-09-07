# EigenPyR

EigenPyR is a tutorial-style project for implementing numerical and statistical learning methods in C++ using the Eigen library and connecting them to Python and R through binding interfaces.

This project focuses on building efficient C++ implementations while keeping the user-facing interface simple and accessible from high-level languages such as Python and R.

## Overview

The main goals of this project are:

- to implement numerical algorithms using Eigen-based C++ code,
- to demonstrate how C++ implementations can be exposed to Python and R,
- to compare computational performance across different implementation settings,
- to provide a compact tutorial-style example of C++ numerical programming and language bindings.

## Implemented Methods

The main methods used in the paper are:

- Kernel Ridge Regression
- Matrix Factorization with Stochastic Gradient Descent

The project also includes additional Eigen-based implementations of basic linear models:

- Ordinary Least Squares Regression
- Ridge Regression
- Lasso Regression

In addition to the core C++ implementations, the project includes:

- Python binding using `pybind11`
- R binding interface
- C++ example files
- Benchmark examples using Eigen and Armadillo

## Project Structure

```text
EigenPyR/
├── core/
│   ├── include/
│   │   ├── linear_models/
│   │   │   ├── lm_ols.h
│   │   │   ├── lm_ridge.h
│   │   │   ├── lm_lasso.h
│   │   │   └── utils.h
│   │   ├── kernel_ridge.h
│   │   └── mf_sgd.h
│   └── src/
│       ├── linear_models/
│       │   ├── lm_ols.cpp
│       │   ├── lm_ridge.cpp
│       │   ├── lm_lasso.cpp
│       │   └── utils.cpp
│       ├── kernel_ridge.cpp
│       └── mf_sgd.cpp
├── bindings/
│   ├── python/
│   │   ├── bindings.cpp
│   │   └── tests/
│   └── r/
├── examples/
│   ├── linear_models/
│   │   ├── lm_ols_ex.cpp
│   │   ├── lm_ridge_ex.cpp
│   │   └── lm_lasso_ex.cpp
│   ├── kernel_ridge_ex.cpp
│   └── mf_sgd_ex.cpp
├── benchmark/
├── external/
├── CMakeLists.txt
└── README.md
```

## Core Implementation

The `core/` directory contains the main Eigen-based C++ implementations.

The primary methods used in the paper are placed directly under `core/include/` and `core/src/`:

```text
kernel_ridge.h / kernel_ridge.cpp
mf_sgd.h / mf_sgd.cpp
```

Additional basic linear model implementations are organized separately under `linear_models/`:

```text
linear_models/
├── lm_ols
├── lm_ridge
├── lm_lasso
└── utils
```

This structure separates the main methods discussed in the paper from additional supporting implementations.

## External Dependencies

This project uses external libraries such as Eigen and pybind11.

The `external/` directory is used to store these third-party libraries locally:

```text
external/
├── eigen/
└── pybind11/
```

In this project, Eigen is used for matrix and vector computations, while pybind11 is used to create Python bindings for the C++ implementations.

The `external/` directory is excluded from Git tracking through `.gitignore`. Therefore, Eigen and pybind11 are not included directly in this repository. Before building the project, users should download these libraries separately and place them inside the `external/` directory.

The expected directory structure is:

```text
EigenPyR/
├── external/
│   ├── eigen/
│   │   └── Eigen/
│   └── pybind11/
│       ├── include/
│       ├── tools/
│       └── CMakeLists.txt
├── CMakeLists.txt
└── ...
```

Eigen is a header-only C++ template library. pybind11 is commonly used through its headers together with its CMake configuration files, so the full pybind11 source directory should be placed under `external/pybind11/` when using `add_subdirectory()` in CMake.

## Requirements

The project uses the following tools and libraries:

- C++20
- CMake
- Eigen
- pybind11
- Rcpp or related R binding tools
- Armadillo, for benchmark comparison

## Build

A typical CMake build can be performed as follows:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

To build only the C++ examples without Python bindings, use:

```bash
cmake .. -DBUILD_PYTHON=OFF
cmake --build .
```

## C++ Examples

Example files are provided under the `examples/` directory.

The main examples used in the paper are:

```text
examples/kernel_ridge_ex.cpp
examples/mf_sgd_ex.cpp
```

Additional examples for basic linear models are provided separately under:

```text
examples/linear_models/
```

## Python Binding

The Python binding source file is located at:

```text
bindings/python/bindings.cpp
```

Python test scripts are organized under:

```text
bindings/python/tests/
```

After building the Python module, the package can be imported in Python as follows:

```python
import eigenPyR
```

