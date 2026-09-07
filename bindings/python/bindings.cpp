#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "linear_models/lm_ols.h"
#include "linear_models/lm_lasso.h"
#include "linear_models/lm_ridge.h"
#include "mf_sgd.h"
#include "kernel_ridge.h"
#include "linear_models/utils.h"

namespace py = pybind11;

static void bind_lm_ols(py::module_& m) {
    py::class_<LmOLS>(m, "LmOLS", R"doc(
Ordinary least squares linear regression.

Fits y = X @ coef + intercept by minimizing the residual sum of squares.
The intercept is derived from the column means, so `X` should not include
an intercept column of its own.
)doc")
        .def(py::init<>(), "Create an unfitted OLS model.")
        .def("fit", &LmOLS::fit, py::arg("X"), py::arg("y"),
             R"doc(
Estimate the regression coefficients from training data.

Parameters
----------
X : numpy.ndarray of shape (n, p)
    Design matrix, without an intercept column.
y : numpy.ndarray of shape (n,)
    Response vector.
)doc")
        .def("predict", &LmOLS::predict, py::arg("X_new"),
             R"doc(
Predict responses for new observations.

Parameters
----------
X_new : numpy.ndarray of shape (m, p)
    Matrix with the same columns as the training data.

Returns
-------
numpy.ndarray of shape (m,)
    Predicted responses.
)doc")
        .def_property_readonly("coef", &LmOLS::coef,
             "Estimated slope coefficients, an array of length p.")
        .def_property_readonly("intercept", &LmOLS::intercept,
             "Estimated intercept term.");
}

static void bind_lm_lasso(py::module_& m) {
    py::class_<LmLasso>(m, "LmLasso", R"doc(
Lasso (L1-penalized) linear regression, fit by coordinate descent.

Minimizes the residual sum of squares plus `lambda` times the L1 norm of
the slope coefficients, which drives some of them exactly to zero.
)doc")
        .def(py::init<double>(), py::arg("lambda"),
             "Create a lasso model with the given L1 penalty strength "
             "(0 reduces to OLS).")
        .def(
            "fit",
            &LmLasso::fit,
            py::arg("X"),
            py::arg("y"),
            py::arg("max_iter") = 500,
            py::arg("eps") = 1e-6,
            R"doc(
Estimate the regression coefficients by coordinate descent.

Parameters
----------
X : numpy.ndarray of shape (n, p)
    Design matrix, without an intercept column.
y : numpy.ndarray of shape (n,)
    Response vector.
max_iter : int, default 500
    Maximum number of coordinate descent sweeps.
eps : float, default 1e-6
    Convergence tolerance on the largest coefficient change.
)doc"
        )
        .def("predict", &LmLasso::predict, py::arg("X_new"),
             R"doc(
Predict responses for new observations.

Parameters
----------
X_new : numpy.ndarray of shape (m, p)
    Matrix with the same columns as the training data.

Returns
-------
numpy.ndarray of shape (m,)
    Predicted responses.
)doc")
        .def_property_readonly("coef", &LmLasso::coef,
             "Estimated slope coefficients; may contain exact zeros.")
        .def_property_readonly("intercept", &LmLasso::intercept,
             "Estimated intercept term.")
        .def_property_readonly("lambda_", &LmLasso::lambda,
             "The L1 penalty strength this model was constructed with.");
}

static void bind_lm_ridge(py::module_& m) {
    py::class_<LmRidge>(m, "LmRidge", R"doc(
Ridge (L2-penalized) linear regression.

Minimizes the residual sum of squares plus `lambda` times the squared L2
norm of the slope coefficients. The intercept is not penalized.
)doc")
        .def(py::init<double>(), py::arg("lambda"),
             "Create a ridge model with the given L2 penalty strength "
             "(0 reduces to OLS).")
        .def("fit", &LmRidge::fit, py::arg("X"), py::arg("y"),
             R"doc(
Estimate the regression coefficients from training data.

Parameters
----------
X : numpy.ndarray of shape (n, p)
    Design matrix, without an intercept column.
y : numpy.ndarray of shape (n,)
    Response vector.
)doc")
        .def("predict", &LmRidge::predict, py::arg("X_new"),
             R"doc(
Predict responses for new observations.

Parameters
----------
X_new : numpy.ndarray of shape (m, p)
    Matrix with the same columns as the training data.

Returns
-------
numpy.ndarray of shape (m,)
    Predicted responses.
)doc")
        .def_property_readonly("coef", &LmRidge::coef,
             "Estimated slope coefficients, an array of length p.")
        .def_property_readonly("intercept", &LmRidge::intercept,
             "Estimated intercept term.")
        .def_property_readonly("lambda_", &LmRidge::lambda,
             "The L2 penalty strength this model was constructed with.");
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
        py::arg("scaling") = true,
        R"doc(
Center and scale the columns of a matrix.

Parameters
----------
X : numpy.ndarray of shape (n, p)
    Matrix to standardize.
centering : bool, default True
    Subtract each column's mean.
scaling : bool, default True
    Divide each column by its standard deviation.

Returns
-------
tuple of (X_std, col_means, col_sds)
)doc"
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
        py::arg("scaling") = true,
        R"doc(
Center and scale a vector.

Parameters
----------
y : numpy.ndarray of shape (n,)
    Vector to standardize.
centering : bool, default True
    Subtract the mean.
scaling : bool, default True
    Divide by the standard deviation.

Returns
-------
tuple of (y_std, mean, sd)
)doc"
    );
}

static void bind_mf_sgd(py::module_& m) {
    py::class_<Rating>(m, "Rating", "A single observed (user, item, value) entry.")
        .def(py::init<>())
        .def_readwrite("user", &Rating::user, "Zero-based user index.")
        .def_readwrite("item", &Rating::item, "Zero-based item index.")
        .def_readwrite("value", &Rating::value, "Observed rating value.");

    py::class_<MatrixFactorizationSGD>(m, "MatrixFactorizationSGD", R"doc(
Matrix factorization trained by stochastic gradient descent.

Approximates the sparse ratings matrix as a global mean plus per-user and
per-item biases plus the inner product of low-rank user/item factor
matrices, trained with L2-regularized SGD over the observed ratings.
)doc")
        .def(
            py::init<int, int, int, double, double, int, unsigned int>(),
            py::arg("n_users"),
            py::arg("n_items"),
            py::arg("n_factors") = 10,
            py::arg("lr") = 0.01,
            py::arg("reg") = 0.02,
            py::arg("n_epochs") = 20,
            py::arg("seed") = 42,
            R"doc(
Create an untrained factorization model.

Parameters
----------
n_users : int
    Number of distinct users; user indices must be below this.
n_items : int
    Number of distinct items; item indices must be below this.
n_factors : int, default 10
    Rank of the latent factor matrices.
lr : float, default 0.01
    SGD learning rate.
reg : float, default 0.02
    L2 regularization strength applied to factors and biases.
n_epochs : int, default 20
    Number of passes over the training ratings.
seed : int, default 42
    Seed for the random number generator used to initialize factors.
)doc"
        )
        .def("fit", &MatrixFactorizationSGD::fit,
             py::arg("ratings"),
             py::arg("verbose") = true,
             R"doc(
Train the factorization on observed ratings.

Parameters
----------
ratings : list of Rating
    Observed ratings to train on.
verbose : bool, default True
    Print the training RMSE after each epoch.
)doc")
        .def("predict", &MatrixFactorizationSGD::predict,
             py::arg("user"),
             py::arg("item"),
             R"doc(
Predict the rating a single user would give a single item.

Parameters
----------
user : int
    Zero-based user index.
item : int
    Zero-based item index.

Returns
-------
float
)doc")
        .def("full_prediction", &MatrixFactorizationSGD::full_prediction,
             "Reconstruct the dense prediction matrix for every user-item pair, "
             "shape (n_users, n_items).")
        .def_property_readonly("user_factors", &MatrixFactorizationSGD::user_factors,
             "Learned user factor matrix, shape (n_users, n_factors).")
        .def_property_readonly("item_factors", &MatrixFactorizationSGD::item_factors,
             "Learned item factor matrix, shape (n_items, n_factors).")
        .def_property_readonly("user_bias", &MatrixFactorizationSGD::user_bias,
             "Learned per-user bias terms, an array of length n_users.")
        .def_property_readonly("item_bias", &MatrixFactorizationSGD::item_bias,
             "Learned per-item bias terms, an array of length n_items.")
        .def_property_readonly("global_mean", &MatrixFactorizationSGD::global_mean,
             "Mean of all training ratings, used as the prediction baseline.");
}

static void bind_kernel_ridge(py::module_& m) {
    py::class_<KernelRidge>(m, "KernelRidge", R"doc(
Kernel ridge regression with a Gaussian (RBF) kernel.

Solves ridge regression in the feature space induced by
k(a, b) = exp(-||a - b||^2 / (2 * sigma^2)), giving the dual solution
alpha = (K + lambda * I)^-1 @ (y - mean(y)). The training inputs are
retained on the fitted object since prediction needs them.
)doc")
        .def(py::init<double, double>(),
             py::arg("lambda_"),
             py::arg("sigma"),
             "Create an unfitted kernel ridge model with the given ridge "
             "penalty and Gaussian kernel bandwidth.")
        .def("fit", &KernelRidge::fit,
             py::arg("X"),
             py::arg("y"),
             R"doc(
Fit the model by solving the regularized dual system.

Parameters
----------
X : numpy.ndarray of shape (n, p)
    Training design matrix.
y : numpy.ndarray of shape (n,)
    Response vector.
)doc")
        .def("predict", &KernelRidge::predict,
             py::arg("X_new"),
             R"doc(
Predict responses for new observations.

Parameters
----------
X_new : numpy.ndarray of shape (m, p)
    Matrix with the same columns as the training data.

Returns
-------
numpy.ndarray of shape (m,)
    Predicted responses.
)doc")
        .def_property_readonly("alpha", &KernelRidge::alpha,
             "Dual coefficients, an array of length n.")
        .def_property_readonly("x_train", &KernelRidge::x_train,
             "Training design matrix retained for prediction.")
        .def_property_readonly("lambda_", &KernelRidge::lambda,
             "The ridge penalty this model was constructed with.")
        .def_property_readonly("sigma", &KernelRidge::sigma,
             "The Gaussian kernel bandwidth this model was constructed with.")
        .def_property_readonly("y_mean", &KernelRidge::y_mean,
             "Mean of the training responses, added back when predicting.");
}

PYBIND11_MODULE(eigenPyR, m) {
    m.doc() = "Python bindings for EigenPyR";
    m.attr("__version__") = "0.1.0";  // keep in sync with pyproject.toml [project].version

    bind_lm_ols(m);
    bind_lm_lasso(m);
    bind_lm_ridge(m);
    bind_utils(m);
    bind_mf_sgd(m);
    bind_kernel_ridge(m);
}
