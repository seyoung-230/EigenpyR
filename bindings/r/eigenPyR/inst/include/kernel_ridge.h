#ifndef SES_KERNEL_RIDGE_H
#define SES_KERNEL_RIDGE_H

#include <Eigen/Dense>
using namespace Eigen;

class KernelRidge {
public:
    KernelRidge(double lambda, double sigma);

    void fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict(const MatrixXd& X_new) const;

    const VectorXd& alpha() const { return alpha_; }
    const MatrixXd& x_train() const { return X_train_; }
    double lambda() const { return lambda_; }
    double sigma() const { return sigma_; }
    double y_mean() const { return y_mean_; }

private:
    MatrixXd kernel_matrix(const MatrixXd& A, const MatrixXd& B) const;

    MatrixXd X_train_;
    VectorXd alpha_;
    double lambda_ = 0.0;
    double sigma_ = 1.0;
    double y_mean_ = 0.0;
};

#endif // SES_KERNEL_RIDGE_H