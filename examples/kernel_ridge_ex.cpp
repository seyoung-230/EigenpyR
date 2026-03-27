#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Dense>
#include "kernel_ridge.h"

using namespace std;

const double PI = 3.14159265358979323846;

double ftrue(double x) {
    return sin(2.0 * PI * x) + 0.5 * cos(4.0 * PI * x);
}

int main() {
    int n = 150;
    MatrixXd X(n, 1);
    VectorXd y(n);

    mt19937 gen(1);
    uniform_real_distribution<double> unif(-1.0, 1.0);
    normal_distribution<double> noise(0.0, 0.1);

    for (int i = 0; i < n; ++i) {
        double x = unif(gen);
        X(i, 0) = x;
        y(i) = ftrue(x) + noise(gen);
    }

    double lambda = 1e-3;
    double sigma = 0.2;

    KernelRidge model(lambda, sigma);
    model.fit(X, y);

    VectorXd fitted = model.predict(X);
    double train_mse = (fitted - y).squaredNorm() / y.size();

    cout << "Kernel Ridge Regression example" << endl;
    cout << "n = " << n << endl;
    cout << "lambda = " << lambda << endl;
    cout << "sigma = " << sigma << endl;
    cout << "y_mean = " << model.y_mean() << endl;
    cout << "Train MSE = " << train_mse << endl;

    int m = 10;
    MatrixXd X_grid(m, 1);

    for (int i = 0; i < m; ++i) {
        X_grid(i, 0) = -1.0 + 2.0 * i / (m - 1.0);
    }

    VectorXd pred = model.predict(X_grid);

    cout << "\nPredictions on grid:" << endl;
    for (int i = 0; i < m; ++i) {
        cout << "x = " << X_grid(i, 0)
             << ", yhat = " << pred(i) << endl;
    }

    return 0;
}