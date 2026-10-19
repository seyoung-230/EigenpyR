cmake_minimum_required(VERSION 3.15)

project(KRRBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------
# Eigen internal engine + Armadillo benchmark
# -------------------------------------------------
add_executable(bench_krr_core
    src/main.cpp
    eigen/kernel_ridge.cpp
    arma/kernel_ridge_arma.cpp
)

target_include_directories(bench_krr_core PRIVATE
    eigen_lib/eigen
    eigen
    arma
)

target_compile_options(bench_krr_core PRIVATE
    -O3
)

target_link_libraries(bench_krr_core PRIVATE
    armadillo
)

# -------------------------------------------------
# Eigen + BLAS + Armadillo benchmark
# -------------------------------------------------
add_executable(bench_krr_blas
    src/main.cpp
    eigen/kernel_ridge.cpp
    arma/kernel_ridge_arma.cpp
)

target_include_directories(bench_krr_blas PRIVATE
    eigen_lib/eigen
    eigen
    arma
)

target_compile_definitions(bench_krr_blas PRIVATE
    EIGEN_USE_BLAS
)

target_compile_options(bench_krr_blas PRIVATE
    -O3
)

target_link_libraries(bench_krr_blas PRIVATE
    openblas
    armadillo
)