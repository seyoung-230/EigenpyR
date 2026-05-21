#include "linear_models/utils.h"
#include "linear_models/lm_ols.h"

void LmOLS::fit(const MatrixXd &X, const VectorXd &y) {
    int n = X.rows();
    int p = X.cols();

    // centering
    VectorXd col_means, col_sds;
    MatrixXd X_std(n, p);
    VectorXd y_std(n);
    double y_mean, y_sd;
    standardization(X, X_std, col_means, col_sds, true, false);
    standardization(y, y_std, y_mean, y_sd, true, false);

    coef_ = (X_std.transpose() * X_std).llt().solve(X_std.transpose() * y_std);
    intercept_ = y_mean - col_means.dot(coef_);
}

VectorXd LmOLS::predict(const MatrixXd& X_new) const {
    return (X_new * coef_).array() + intercept_;
}
