# EigenPyR

EigenPyR is a tutorial-style project for implementing numerical methods in C++ using the Eigen library and connecting them to Python and R through binding interfaces.

This project focuses on building efficient C++ implementations while keeping the user-facing interface simple and accessible from high-level languages such as Python and R.

## Overview

The main goals of this project are:

- to implement numerical algorithms using Eigen-based C++ code,
- to demonstrate how C++ implementations can be exposed to Python and R,
- to compare computational performance across different implementation settings,
- to provide a simple tutorial-style example of C++ numerical programming and language bindings.

## Implemented Methods

Currently, the project includes:

- Kernel Ridge Regression
- Matrix Factorization with Stochastic Gradient Descent
- Python binding using `pybind11`
- R binding interface
- Benchmark examples using Eigen and Armadillo

## Project Structure

```text
EigenPyR/
├── core/
│   ├── include/
│   └── src/
├── bindings/
│   ├── python/
├── examples/
├── benchmark/
├── external/
├── CMakeLists.txt
└── README.md

