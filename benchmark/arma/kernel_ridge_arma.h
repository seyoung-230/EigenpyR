#ifndef SES_KERNEL_RIDGE_ARMA_H
#define SES_KERNEL_RIDGE_ARMA_H

#include <armadillo>

class KernelRidgeArma {
public:
    KernelRidgeArma(double lambda, double sigma);

    void fit(const arma::mat& X, const arma::vec& y);
    arma::vec predict(const arma::mat& X_new) const;

private:
    arma::mat kernel_matrix(const arma::mat& A, const arma::mat& B) const;

    arma::mat X_train_;
    arma::vec alpha_;

    double lambda_;
    double sigma_;
    double y_mean_;
};

#endif