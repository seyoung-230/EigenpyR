#include <Eigen/Dense>
#include <iostream>
#include "utils.h"

using namespace Eigen;
using namespace std;

int main() {
    MatrixXd X(3, 3);
    X << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    MatrixXd X_std(3, 3);
    VectorXd col_means, col_sds;


    standardization(X, X_std, col_means, col_sds);

    cout << col_means << endl << endl;
    cout << col_sds << endl << endl;
    cout << X << endl << endl;
    cout << X_std << endl << endl;

    VectorXd y(3);
    y << 1, 2, 3;
    VectorXd y_std(3);
    double col_means_y, col_sds_y;

    standardization(y, y_std, col_means_y, col_sds_y);

    cout << col_means_y << endl << endl;
    cout << col_sds_y << endl << endl;
    cout << y << endl << endl;
    cout<< y_std << endl << endl;

    return 0;
}