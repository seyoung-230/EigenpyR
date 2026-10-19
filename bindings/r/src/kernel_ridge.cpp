#include "kernel_ridge.h"

#include <cmath>
#include <stdexcept>

KernelRidge::KernelRidge(double lambda, double sigma)
    : lambda_(lambda), sigma_(sigma) {
    if (lambda_ < 0.0) {
        throw std::invalid_argument("lambda must be non-negative.");
    }
    if (sigma_ <= 0.0) {
        throw std::invalid_argument("sigma must be positive.");
    }
}

MatrixXd KernelRidge::kernel_matrix(const MatrixXd& A, const MatrixXd& B) const {
    if (A.cols() != B.cols()) {
        throw std::invalid_argument("A and B must have the same number of columns.");
    }

    int n1 = static_cast<int>(A.rows());
    int n2 = static_cast<int>(B.rows());
    MatrixXd K(n1, n2);

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            double dist2 = (A.row(i) - B.row(j)).squaredNorm();
            K(i, j) = std::exp(-dist2 / (2.0 * sigma_ * sigma_));
        }
    }

    return K;
}

void KernelRidge::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("Number of rows in X must match length of y.");
    }

    X_train_ = X;

    y_mean_ = y.mean();
    VectorXd y_centered = y.array() - y_mean_;

    int n = static_cast<int>(X.rows());
    MatrixXd K = kernel_matrix(X_train_, X_train_);
    MatrixXd I = MatrixXd::Identity(n, n);

    alpha_ = (K + lambda_ * I).ldlt().solve(y_centered);
}

VectorXd KernelRidge::predict(const MatrixXd& X_new) const {
    if (X_train_.rows() == 0 || alpha_.size() == 0) {
        throw std::runtime_error("Model has not been fitted yet.");
    }

    MatrixXd K_new = kernel_matrix(X_new, X_train_);
    VectorXd pred = K_new * alpha_;
    pred.array() += y_mean_;

    return pred;
}