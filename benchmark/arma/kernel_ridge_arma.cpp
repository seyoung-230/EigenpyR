#include "kernel_ridge_arma.h"

#include <cmath>
#include <stdexcept>

KernelRidgeArma::KernelRidgeArma(double lambda, double sigma)
    : lambda_(lambda), sigma_(sigma), y_mean_(0.0) {
    if (lambda_ < 0.0) {
        throw std::invalid_argument("lambda must be non-negative.");
    }
    if (sigma_ <= 0.0) {
        throw std::invalid_argument("sigma must be positive.");
    }
}

arma::mat KernelRidgeArma::kernel_matrix(
    const arma::mat& A,
    const arma::mat& B
) const {
    if (A.n_cols != B.n_cols) {
        throw std::invalid_argument(
            "A and B must have the same number of columns."
        );
    }

    arma::uword n1 = A.n_rows;
    arma::uword n2 = B.n_rows;
    arma::mat K(n1, n2);

    for (arma::uword i = 0; i < n1; ++i) {
        for (arma::uword j = 0; j < n2; ++j) {
            double dist2 = arma::accu(
                arma::square(A.row(i) - B.row(j))
            );

            K(i, j) = std::exp(
                -dist2 / (2.0 * sigma_ * sigma_)
            );
        }
    }

    return K;
}

void KernelRidgeArma::fit(const arma::mat& X, const arma::vec& y) {
    if (X.n_rows != y.n_elem) {
        throw std::invalid_argument(
            "Number of rows in X must match length of y."
        );
    }

    X_train_ = X;

    y_mean_ = arma::mean(y);
    arma::vec y_centered = y - y_mean_;

    arma::uword n = X.n_rows;
    arma::mat K = kernel_matrix(X_train_, X_train_);
    K.diag() += lambda_;

    alpha_ = arma::solve(K, y_centered);
}

arma::vec KernelRidgeArma::predict(const arma::mat& X_new) const {
    if (X_train_.n_rows == 0 || alpha_.n_elem == 0) {
        throw std::runtime_error("Model has not been fitted yet.");
    }

    arma::mat K_new = kernel_matrix(X_new, X_train_);
    arma::vec pred = K_new * alpha_;
    pred += y_mean_;

    return pred;
}