#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "lm_ols.h"
#include "lm_lasso.h"
#include "lm_ridge.h"
#include "mf_sgd.h"
#include "kernel_ridge.h"
#include "utils.h"

namespace py = pybind11;

static void bind_lm_ols(py::module_& m) {
    py::class_<LmOLS>(m, "LmOLS")
        .def(py::init<>())
        .def("fit", &LmOLS::fit, py::arg("X"), py::arg("y"))
        .def("predict", &LmOLS::predict, py::arg("X_new"))
        .def_property_readonly("coef", &LmOLS::coef)
        .def_property_readonly("intercept", &LmOLS::intercept);
}

static void bind_lm_lasso(py::module_& m) {
    py::class_<LmLasso>(m, "LmLasso")
        .def(py::init<double>(), py::arg("lambda"))
        .def(
            "fit",
            &LmLasso::fit,
            py::arg("X"),
            py::arg("y"),
            py::arg("max_iter") = 500,
            py::arg("eps") = 1e-6
        )
        .def("predict", &LmLasso::predict, py::arg("X_new"))
        .def_property_readonly("coef", &LmLasso::coef)
        .def_property_readonly("intercept", &LmLasso::intercept)
        .def_property_readonly("lambda_", &LmLasso::lambda);
}

static void bind_lm_ridge(py::module_& m) {
    py::class_<LmRidge>(m, "LmRidge")
        .def(py::init<double>(), py::arg("lambda"))
        .def("fit", &LmRidge::fit, py::arg("X"), py::arg("y"))
        .def("predict", &LmRidge::predict, py::arg("X_new"))
        .def_property_readonly("coef", &LmRidge::coef)
        .def_property_readonly("intercept", &LmRidge::intercept)
        .def_property_readonly("lambda_", &LmRidge::lambda);
}

static void bind_utils(py::module_& m) {
    m.def(
        "standardize_matrix",
        [](const Eigen::MatrixXd& X, bool centering, bool scaling) {
            Eigen::MatrixXd X_std;
            Eigen::VectorXd col_means;
            Eigen::VectorXd col_sds;
            standardization(X, X_std, col_means, col_sds, centering, scaling);
            return py::make_tuple(X_std, col_means, col_sds);
        },
        py::arg("X"),
        py::arg("centering") = true,
        py::arg("scaling") = true
    );

    m.def(
        "standardize_vector",
        [](const Eigen::VectorXd& y, bool centering, bool scaling) {
            Eigen::VectorXd y_std;
            double mean;
            double sd;
            standardization(y, y_std, mean, sd, centering, scaling);
            return py::make_tuple(y_std, mean, sd);
        },
        py::arg("y"),
        py::arg("centering") = true,
        py::arg("scaling") = true
    );
}

static void bind_mf_sgd(py::module_& m) {
    py::class_<Rating>(m, "Rating")
        .def(py::init<>())
        .def_readwrite("user", &Rating::user)
        .def_readwrite("item", &Rating::item)
        .def_readwrite("value", &Rating::value);

    py::class_<MatrixFactorizationSGD>(m, "MatrixFactorizationSGD")
        .def(
            py::init<int, int, int, double, double, int, unsigned int>(),
            py::arg("n_users"),
            py::arg("n_items"),
            py::arg("n_factors") = 10,
            py::arg("lr") = 0.01,
            py::arg("reg") = 0.02,
            py::arg("n_epochs") = 20,
            py::arg("seed") = 42
        )
        .def("fit", &MatrixFactorizationSGD::fit,
             py::arg("ratings"),
             py::arg("verbose") = true)
        .def("predict", &MatrixFactorizationSGD::predict,
             py::arg("user"),
             py::arg("item"))
        .def("full_prediction", &MatrixFactorizationSGD::full_prediction)
        .def_property_readonly("user_factors", &MatrixFactorizationSGD::user_factors)
        .def_property_readonly("item_factors", &MatrixFactorizationSGD::item_factors)
        .def_property_readonly("user_bias", &MatrixFactorizationSGD::user_bias)
        .def_property_readonly("item_bias", &MatrixFactorizationSGD::item_bias)
        .def_property_readonly("global_mean", &MatrixFactorizationSGD::global_mean);
}

static void bind_kernel_ridge(py::module_& m) {
    py::class_<KernelRidge>(m, "KernelRidge")
        .def(py::init<double, double>(),
             py::arg("lambda"),
             py::arg("sigma"))
        .def("fit", &KernelRidge::fit,
             py::arg("X"),
             py::arg("y"))
        .def("predict", &KernelRidge::predict,
             py::arg("X_new"))
        .def_property_readonly("alpha", &KernelRidge::alpha)
        .def_property_readonly("x_train", &KernelRidge::x_train)
        .def_property_readonly("lambda_", &KernelRidge::lambda)
        .def_property_readonly("sigma", &KernelRidge::sigma)
        .def_property_readonly("y_mean", &KernelRidge::y_mean);
}

PYBIND11_MODULE(eigenpyr, m) {
    m.doc() = "Python bindings for EigenPyR";

    bind_lm_ols(m);
    bind_lm_lasso(m);
    bind_lm_ridge(m);
    bind_utils(m);
    bind_mf_sgd(m);
    bind_kernel_ridge(m);
}