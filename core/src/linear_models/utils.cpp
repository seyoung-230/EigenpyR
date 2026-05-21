#include "linear_models/utils.h"

void standardization(const MatrixXd& X, MatrixXd& X_std, VectorXd& col_means, VectorXd& col_sds, bool centering, bool scaling) {
    int n = X.rows();
    int p = X.cols();
    col_means = X.colwise().mean();
    MatrixXd centered_X = X.rowwise() - col_means.transpose();
    col_sds = (centered_X.array().square().colwise().sum() / n).sqrt();

    if (centering) {
        if (scaling) {
            X_std = centered_X.array().rowwise() / col_sds.transpose().array();
        } else {
            X_std = centered_X;
        }
    }
    if (!centering && scaling) {throw std::runtime_error("scaling without centering is not supported.");}
    if (!centering && !scaling) { X_std = X;}
}

void standardization(const VectorXd& y, VectorXd& y_std, double& mean, double& sd, bool centering, bool scaling) {
    int n = y.size();
    mean = y.mean();
    VectorXd centered_y = y.array() - mean;
    sd = sqrt(centered_y.array().square().sum() / n);
    if (centering) {
        if (scaling) {
            y_std = centered_y.array() / sd;
        } else {
            y_std = centered_y;
        }
    }
    if (!centering && scaling) {throw std::runtime_error("scaling without centering is not supported.");}
    if (!centering && !scaling) { y_std = y;}
}