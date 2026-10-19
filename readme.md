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

- **Eigen** (header-only, MPL2) is vendored under `external/eigen/` (version 5.0.1), so no separate installation is needed.
- **pybind11** is not vendored; it is obtained from the `pybind11` package on PyPI at build time (declared in `pyproject.toml`). For a plain CMake build outside `pip`, run `pip install pybind11` first.
- **Rcpp** and **RcppEigen** (R packages) are required for the R binding.
- **Armadillo** and **OpenBLAS** are required only for the benchmark in `benchmark/`.
## Requirements

The project uses the following tools and libraries:

- C++20
- CMake
- Eigen (vendored)
- pybind11 (from PyPI)
- Rcpp and RcppEigen (R binding)
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

## R Binding

The R package `eigenPyR` is located directly under `bindings/r/`. Before compilation, its `configure` script copies the sources from `core/` into the package, so `core/` remains the single implementation that both bindings use. Install it from the repository root with

```bash
R CMD INSTALL bindings/r
```

and use it from R as follows:

```r
library(eigenPyR)
model <- new(KernelRidge, 0.001, 0.2)
X <- matrix(seq(-1, 1, length.out = 100), ncol = 1)
model$fit(X, sin(2 * pi * X[, 1]))
pred <- model$predict(X)
```

## Benchmark

The kernel ridge regression benchmark reported in the paper is located in `benchmark/`. It compares Eigen with its internal engine, Eigen linked to OpenBLAS (`EIGEN_USE_BLAS`), and Armadillo. It requires Armadillo and OpenBLAS; the Eigen 5.0.0 headers used for the reported timings are included in `benchmark/eigen_lib/`.

```bash
cd benchmark
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bench_krr_core   # Eigen internal engine vs Armadillo; writes result.csv
./build/bench_krr_blas   # Eigen linked to OpenBLAS vs Armadillo; writes result.csv
```

Each program writes `result.csv` to the current directory, so rename it after each run. The settings (`d = 5`, `n_test = 100`, 15 repetitions, `n = 1000, ..., 4000`) are set in `benchmark/src/main.cpp`. The files `result_default.csv` and `result_blas.csv` contain the timings reported in the paper.