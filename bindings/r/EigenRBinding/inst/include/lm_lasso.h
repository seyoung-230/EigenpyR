#ifndef SES_LM_LASSO_H
#define SES_LM_LASSO_H

#include <Eigen/Dense>
using namespace Eigen;

class LmLasso {
public:
    explicit LmLasso(double lambda);
    void fit(const MatrixXd& X, const VectorXd& y, int max_iter = 500, double eps = 1e-6);
    VectorXd predict(const MatrixXd& X_new) const;
    VectorXd coef() const { return coef_;}
    double intercept() const { return intercept_;}
    double lambda() {return lambda_;}
private:
    VectorXd coef_;
    double intercept_;
    double lambda_ = 0.0;
};

double soft_threshold(double x, double lambda);

#endif //SES_LM_LASSO_H