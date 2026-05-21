#include "linear_models/utils.h"
#include "linear_models/lm_lasso.h"

LmLasso::LmLasso(double lambda) {
    lambda_ = lambda;
}

void LmLasso::fit(const MatrixXd &X, const VectorXd &y, int max_iter, double eps) {
    int n = X.rows();
    int p = X.cols();

    // standardization
    VectorXd col_means, col_sds;
    MatrixXd X_std(n, p);
    VectorXd y_std(n);
    double y_mean, y_sd;
    standardization(X, X_std, col_means, col_sds, true, true);
    standardization(y, y_std, y_mean, y_sd, true, false);

    // initialization
    VectorXd coef = VectorXd::Zero(p);
    VectorXd coef_old;
    VectorXd residual = y_std;
    double diff = 1.0;


    for (int iter = 0; iter < max_iter; iter++) {
        coef_old = coef;
        for (int j = 0; j < p; j++) {
            residual += X_std.col(j) * coef(j);
            coef(j) = soft_threshold((residual.array() * X_std.col(j).array()).sum() / n, lambda_);
            residual -= X_std.col(j) * coef(j);
        }
        diff = (coef - coef_old).cwiseAbs().maxCoeff();
        if (diff < eps)
            break;
    }

    coef_ = coef.array() / col_sds.array();
    intercept_ = y_mean - col_means.dot(coef_);
}

VectorXd LmLasso::predict(const MatrixXd& X_new) const {
    return (X_new * coef_).array() + intercept_;
}

double soft_threshold(double x, double lambda) {
    if (x > lambda) {
        return x - lambda;
    } else if (x < -lambda) {
        return x + lambda;
    } else
        return 0;
}