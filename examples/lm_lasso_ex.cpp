#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "utils.h"
#include "lm_lasso.h"
using namespace std;

int main() {
    int n = 100;
    int p = 5;
    VectorXd true_coef(p + 1);
    true_coef << 1, 2, -1, 0, 0, 0;
    MatrixXd X(n, p);

    std::mt19937 gen(1);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = U(gen);

    std::normal_distribution<double> N01(0.0, 1.0);
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        y(i) = true_coef(0) + 0.1 * N01(gen);
        for (int j = 0; j < p; ++j) {
            y(i) += X(i, j) * true_coef(j + 1);
        }
    }

    LmLasso lm_lasso(0.03);
    lm_lasso.fit(X, y);
    cout << "true coefficients: " << true_coef.transpose() << endl << endl;
    cout << "estimated coefficients: " << lm_lasso.intercept() << " " << lm_lasso.coef().transpose() << endl << endl;

    MatrixXd X_new(3, p);
    X_new << 1, 1, 2, -1, 0,
             3, 1, 2, 1, 3,
             4, 1, 1, 3, 1;
    cout << "new data:\n" << X_new << endl;
    cout << "predicted_y: " << lm_lasso.predict(X_new).transpose() << endl;

    return 0;
}