#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <stdexcept>

struct Rating {
    int user;
    int item;
    double value;
};

class MatrixFactorizationSGD {
public:
    MatrixFactorizationSGD(
        int n_users,
        int n_items,
        int n_factors = 10,
        double lr = 0.01,
        double reg = 0.02,
        int n_epochs = 20,
        unsigned int seed = 42
    );

    void fit(const std::vector<Rating>& ratings, bool verbose = true);

    double predict(int user, int item) const;
    Eigen::MatrixXd full_prediction() const;

    const Eigen::MatrixXd& user_factors() const;
    const Eigen::MatrixXd& item_factors() const;
    const Eigen::VectorXd& user_bias() const;
    const Eigen::VectorXd& item_bias() const;
    double global_mean() const;

private:
    int n_users_;
    int n_items_;
    int n_factors_;
    double lr_;
    double reg_;
    int n_epochs_;
    unsigned int seed_;
    double global_mean_;

    Eigen::MatrixXd P_;
    Eigen::MatrixXd Q_;
    Eigen::VectorXd bu_;
    Eigen::VectorXd bi_;

    std::mt19937 rng_;

    void initialize_parameters();
    void validate_indices(const std::vector<Rating>& ratings) const;
    double compute_rmse(const std::vector<Rating>& ratings) const;
};