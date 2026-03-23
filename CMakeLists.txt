cmake_minimum_required(VERSION 4.0)
project(EigenPyR)

set(CMAKE_CXX_STANDARD 20)

# find_package(Eigen3 REQUIRED)  --> local installation of Eigen not required because of vendoring

# 코어 라이브러리 (헤더/구현)
add_library(eigenpyr_core STATIC
        core/src/lm_ols.cpp
        core/src/utils.cpp
        core/src/lm_lasso.cpp
        core/src/lm_ridge.cpp
        core/src/mf_sgd.cpp
)

target_include_directories(eigenpyr_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)


# examples executable
file(GLOB EXAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/examples/*.cpp")

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