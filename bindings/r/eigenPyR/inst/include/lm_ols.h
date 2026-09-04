#ifndef SES_LM_OLS_H
#define SES_LM_OLS_H

#include <Eigen/Dense>
using namespace Eigen;

class LmOLS {
public:
    LmOLS() {intercept_ = 0.0;}
    void fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict(const MatrixXd& X_new) const;
    VectorXd coef() const { return coef_;}
    double intercept() const { return intercept_;}
private:
    double intercept_;
    VectorXd coef_;
};

#endif //SES_LM_OLS_H