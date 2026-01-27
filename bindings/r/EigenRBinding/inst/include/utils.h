#ifndef SES_UTILS_H
#define SES_UTILS_H

#include <Eigen/Dense>
#include <stdexcept>
using namespace Eigen;

void standardization(const MatrixXd& X, MatrixXd& X_std, VectorXd& col_means, VectorXd& col_sds, bool centering = true, bool scaling = true);
void standardization(const VectorXd& y, VectorXd& y_std, double& mean, double& sd, bool centering = true, bool scaling = true);


#endif //SES_UTILS_H