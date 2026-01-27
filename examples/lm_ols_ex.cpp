#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "utils.h"
#include "lm_ols.h"
using namespace std;

int main() {
    int n = 50;
    int p = 2;
    VectorXd true_coef(p + 1);
    true_coef << 1, 2, -1;
    MatrixXd X(n, p);

    std::mt19937 gen(123);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            X(i, j) = U(gen);

    std::normal_distribution<double> N01(0.0, 1.0);
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        y(i) = true_coef(0) + X(i, 0) * true_coef(1) + X(i, 1) * true_coef(2) + 0.1 * N01(gen);
    }

    LmOLS lm_ols;
    lm_ols.fit(X, y);
    cout << "true coefficients: " << true_coef.transpose() << endl << endl;
    cout << "estimated coefficients: " << lm_ols.intercept() << " " << lm_ols.coef().transpose() << endl << endl;

    MatrixXd X_new(3, p);
    X_new << 1, 1, 2, -1, 0, 3;
    cout << "new data:\n" << X_new << endl;
    cout << "predicted_y: " << lm_ols.predict(X_new).transpose() << endl;

    return 0;
}