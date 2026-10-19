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

    VectorXd A_norm = A.rowwise().squaredNorm();
    VectorXd B_norm = B.rowwise().squaredNorm();

    MatrixXd dist2 =
        A_norm.replicate(1, B.rows()) +
        B_norm.transpose().replicate(A.rows(), 1) -
        2.0 * (A * B.transpose());

    dist2 = dist2.cwiseMax(0.0);

    MatrixXd K = (-dist2.array() / (2.0 * sigma_ * sigma_)).exp();

    return K;
}

void KernelRidge::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("Number of rows in X must match length of y.");
    }

    X_train_ = X;

    y_mean_ = y.mean();
    VectorXd y_centered = y.array() - y_mean_;

    MatrixXd K = kernel_matrix(X_train_, X_train_);

    // Add lambda to the diagonal only instead of forming K + lambda * I explicitly
    K.diagonal().array() += lambda_;

    // In KRR, K + lambda I is symmetric positive definite, so LLT can be used
    alpha_ = K.llt().solve(y_centered);
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