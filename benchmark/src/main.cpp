#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <vector>
#include <fstream>   // file output

#include "../eigen/kernel_ridge.h"
#include "../arma/kernel_ridge_arma.h"

#include <Eigen/Dense>
#include <armadillo>

using namespace std;
using namespace Eigen;

double elapsed_seconds(
    chrono::high_resolution_clock::time_point start,
    chrono::high_resolution_clock::time_point end
) {
    return chrono::duration<double>(end - start).count();
}

int main() {

#ifdef EIGEN_USE_BLAS
    cout << "[Eigen + BLAS build]\n";
#else
    cout << "[Eigen default build]\n";
#endif

    vector<int> n_list = {1000, 2000, 3000, 4000};

    int d = 5;
    int n_test = 100;
    int repeat = 15;

    double lambda = 1e-4;
    double sigma = 1.0;

    mt19937 gen(230);
    normal_distribution<double> dist(0.0, 1.0);

    // create the CSV file
    ofstream fout("result.csv");
    fout << "n,method,fit_time,predict_time\n";

    cout << "\nKRR Benchmark (avg over " << repeat << " runs)\n\n";

    cout << left
         << setw(8)  << "n"
         << setw(15) << "Method"
         << setw(15) << "Fit Time"
         << setw(15) << "Predict Time"
         << endl;

    cout << string(55, '-') << endl;

    for (int n : n_list) {

        MatrixXd X(n, d);
        VectorXd y(n);
        MatrixXd X_test(n_test, d);

        for (int i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < d; ++j) {
                X(i, j) = dist(gen);
                row_sum += X(i, j);
            }
            y(i) = sin(row_sum) + 0.1 * dist(gen);
        }

        for (int i = 0; i < n_test; ++i) {
            for (int j = 0; j < d; ++j) {
                X_test(i, j) = dist(gen);
            }
        }

        arma::mat X_arma(n, d);
        arma::vec y_arma(n);
        arma::mat X_test_arma(n_test, d);

        for (int i = 0; i < n; ++i) {
            y_arma(i) = y(i);
            for (int j = 0; j < d; ++j) {
                X_arma(i, j) = X(i, j);
            }
        }

        for (int i = 0; i < n_test; ++i) {
            for (int j = 0; j < d; ++j) {
                X_test_arma(i, j) = X_test(i, j);
            }
        }

        double eigen_fit = 0, eigen_pred = 0;
        double arma_fit = 0, arma_pred = 0;

        for (int r = 0; r < repeat; ++r) {

            KernelRidge eigen_model(lambda, sigma);

            auto start = chrono::high_resolution_clock::now();
            eigen_model.fit(X, y);
            auto end = chrono::high_resolution_clock::now();
            eigen_fit += elapsed_seconds(start, end);

            start = chrono::high_resolution_clock::now();
            eigen_model.predict(X_test);
            end = chrono::high_resolution_clock::now();
            eigen_pred += elapsed_seconds(start, end);

            KernelRidgeArma arma_model(lambda, sigma);

            start = chrono::high_resolution_clock::now();
            arma_model.fit(X_arma, y_arma);
            end = chrono::high_resolution_clock::now();
            arma_fit += elapsed_seconds(start, end);

            start = chrono::high_resolution_clock::now();
            arma_model.predict(X_test_arma);
            end = chrono::high_resolution_clock::now();
            arma_pred += elapsed_seconds(start, end);
        }

        eigen_fit /= repeat;
        eigen_pred /= repeat;
        arma_fit /= repeat;
        arma_pred /= repeat;

        // console output
        cout << setw(8)  << n
             << setw(15) << "Eigen"
             << setw(15) << eigen_fit
             << setw(15) << eigen_pred << endl;

        cout << setw(8)  << n
             << setw(15) << "Armadillo"
             << setw(15) << arma_fit
             << setw(15) << arma_pred << endl;

        cout << string(55, '-') << endl;

        // save to CSV
        fout << n << ",Eigen," << eigen_fit << "," << eigen_pred << "\n";
        fout << n << ",Armadillo," << arma_fit << "," << arma_pred << "\n";
    }

    fout.close();

    cout << "\nCSV saved to result.csv\n";

    return 0;
}