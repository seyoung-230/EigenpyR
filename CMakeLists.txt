cmake_minimum_required(VERSION 4.0)
project(EigenPyR)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build C++ examples" ON)

# find_package(Eigen3 REQUIRED)  --> local installation of Eigen not required; external/eigen/Eigen is vendored in git

# 코어 라이브러리 (헤더/구현)
add_library(eigenpyr_core STATIC
        core/src/linear_models/lm_ols.cpp
        core/src/linear_models/utils.cpp
        core/src/linear_models/lm_lasso.cpp
        core/src/linear_models/lm_ridge.cpp
        core/src/mf_sgd.cpp
        core/src/kernel_ridge.cpp
)

target_include_directories(eigenpyr_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

# Python binding
if(BUILD_PYTHON)
    # pybind11 comes from the `pybind11` PyPI package (declared in pyproject.toml's
    # build-system requires), not from a vendored copy - scikit-build-core points
    # CMAKE_PREFIX_PATH at it automatically. For a plain `cmake` build outside pip,
    # install pybind11 (`pip install pybind11`) so this find_package can see it.
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(eigenPyR
        bindings/python/bindings.cpp
    )

    target_link_libraries(eigenPyR PRIVATE eigenpyr_core)

    # scikit-build-core installs whatever lands here straight into the wheel root.
    install(TARGETS eigenPyR LIBRARY DESTINATION .)
endif()

# examples executable
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/examples/*.cpp"
                              "${CMAKE_SOURCE_DIR}/examples/linear_models/*.cpp"
    )
    # 각 파일에 대해 반복
    foreach(example_src ${EXAMPLE_SOURCES})
        # 파일 이름에서 확장자 제거
        get_filename_component(example_name ${example_src} NAME_WE)
        # 실행파일 생성
        add_executable(${example_name} ${example_src})
        # ses_core 라이브러리 링크
        target_link_libraries(${example_name} PRIVATE eigenpyr_core)

        message(STATUS "Example added: ${example_name}")
    endforeach()
endif()