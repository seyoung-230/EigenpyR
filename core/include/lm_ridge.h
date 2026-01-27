#ifndef SES_LM_RIDGE_H
#define SES_LM_RIDGE_H

#include <Eigen/Dense>
using namespace Eigen;

class LmRidge {
public:
    explicit LmRidge(double lambda);
    void fit(const MatrixXd& X, const VectorXd& y);
    VectorXd predict(const MatrixXd& X_new) const;
    VectorXd coef() const { return coef_;}
    double intercept() const { return intercept_;}
    double lambda() {return lambda_;}
private:
    VectorXd coef_;
    double intercept_ = 0.0;
    double lambda_ = 0.0;
};

#endif //SES_LM_RIDGE_H