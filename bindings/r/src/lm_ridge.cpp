#include "utils.h"
#include "lm_ridge.h"

LmRidge::LmRidge(double lambda) {
    lambda_ = lambda;
}

void LmRidge::fit(const MatrixXd& X, const VectorXd& y) {
    int n = X.rows();
    int p = X.cols();

    // standardization
    VectorXd col_means, col_sds;
    MatrixXd X_std(n, p);
    VectorXd y_std(n);
    double y_mean, y_sd;
    standardization(X, X_std, col_means, col_sds, true, true);
    standardization(y, y_std, y_mean, y_sd, true, false);

    coef_ = (X_std.transpose() * X_std + lambda_ * MatrixXd::Identity(p, p)).llt().solve(X_std.transpose() * y_std).array() / col_sds.array();
    intercept_ = y_mean - col_means.dot(coef_);
}

VectorXd LmRidge::predict(const MatrixXd& X_new) const {
    return (X_new * coef_).array() + intercept_;
}

