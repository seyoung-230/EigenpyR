#include "mf_sgd.h"

#include <algorithm>
#include <iostream>
#include <cmath>

MatrixFactorizationSGD::MatrixFactorizationSGD(
    int n_users,
    int n_items,
    int n_factors,
    double lr,
    double reg,
    int n_epochs,
    unsigned int seed
)
    : n_users_(n_users),
      n_items_(n_items),
      n_factors_(n_factors),
      lr_(lr),
      reg_(reg),
      n_epochs_(n_epochs),
      seed_(seed),
      global_mean_(0.0),
      P_(n_users, n_factors),
      Q_(n_items, n_factors),
      bu_(n_users),
      bi_(n_items),
      rng_(seed) {

    if (n_users_ <= 0 || n_items_ <= 0 || n_factors_ <= 0) {
        throw std::invalid_argument("n_users, n_items, n_factors must be positive.");
    }
    if (lr_ <= 0.0) {
        throw std::invalid_argument("learning rate must be positive.");
    }
    if (reg_ < 0.0) {
        throw std::invalid_argument("regularization must be non-negative.");
    }
    if (n_epochs_ <= 0) {
        throw std::invalid_argument("n_epochs must be positive.");
    }

    initialize_parameters();
}

void MatrixFactorizationSGD::initialize_parameters() {
    std::normal_distribution<double> dist(0.0, 0.1);

    for (int u = 0; u < n_users_; ++u) {
        for (int k = 0; k < n_factors_; ++k) {
            P_(u, k) = dist(rng_);
        }
    }

    for (int i = 0; i < n_items_; ++i) {
        for (int k = 0; k < n_factors_; ++k) {
            Q_(i, k) = dist(rng_);
        }
    }

    bu_.setZero();
    bi_.setZero();
}

void MatrixFactorizationSGD::validate_indices(const std::vector<Rating>& ratings) const {
    for (const auto& r : ratings) {
        if (r.user < 0 || r.user >= n_users_) {
            throw std::out_of_range("user index out of range.");
        }
        if (r.item < 0 || r.item >= n_items_) {
            throw std::out_of_range("item index out of range.");
        }
    }
}

void MatrixFactorizationSGD::fit(const std::vector<Rating>& ratings, bool verbose) {
    if (ratings.empty()) {
        throw std::invalid_argument("ratings must not be empty.");
    }

    validate_indices(ratings);

    global_mean_ = 0.0;
    for (const auto& r : ratings) {
        global_mean_ += r.value;
    }
    global_mean_ /= static_cast<double>(ratings.size());

    std::vector<Rating> shuffled = ratings;

    for (int epoch = 0; epoch < n_epochs_; ++epoch) {
        std::shuffle(shuffled.begin(), shuffled.end(), rng_);

        for (const auto& r : shuffled) {
            int u = r.user;
            int i = r.item;
            double y = r.value;

            double pred = global_mean_ + bu_(u) + bi_(i) + P_.row(u).dot(Q_.row(i));
            double err = y - pred;

            Eigen::RowVectorXd pu = P_.row(u);
            Eigen::RowVectorXd qi = Q_.row(i);

            bu_(u) += lr_ * (err - reg_ * bu_(u));
            bi_(i) += lr_ * (err - reg_ * bi_(i));

            P_.row(u) += lr_ * (err * qi - reg_ * pu);
            Q_.row(i) += lr_ * (err * pu - reg_ * qi);
        }

        if (verbose) {
            double rmse = compute_rmse(ratings);
            std::cout << "[Epoch " << (epoch + 1) << "/" << n_epochs_
                      << "] RMSE = " << rmse << std::endl;
        }
    }
}

double MatrixFactorizationSGD::predict(int user, int item) const {
    if (user < 0 || user >= n_users_) {
        throw std::out_of_range("user index out of range.");
    }
    if (item < 0 || item >= n_items_) {
        throw std::out_of_range("item index out of range.");
    }

    return global_mean_ + bu_(user) + bi_(item) + P_.row(user).dot(Q_.row(item));
}

Eigen::MatrixXd MatrixFactorizationSGD::full_prediction() const {
    Eigen::MatrixXd pred =
        Eigen::MatrixXd::Constant(n_users_, n_items_, global_mean_);
    pred.colwise() += bu_;
    pred.rowwise() += bi_.transpose();
    pred += P_ * Q_.transpose();
    return pred;
}

double MatrixFactorizationSGD::compute_rmse(const std::vector<Rating>& ratings) const {
    double sse = 0.0;
    for (const auto& r : ratings) {
        double err = r.value - predict(r.user, r.item);
        sse += err * err;
    }
    return std::sqrt(sse / static_cast<double>(ratings.size()));
}

const Eigen::MatrixXd& MatrixFactorizationSGD::user_factors() const {
    return P_;
}

const Eigen::MatrixXd& MatrixFactorizationSGD::item_factors() const {
    return Q_;
}

const Eigen::VectorXd& MatrixFactorizationSGD::user_bias() const {
    return bu_;
}

const Eigen::VectorXd& MatrixFactorizationSGD::item_bias() const {
    return bi_;
}

double MatrixFactorizationSGD::global_mean() const {
    return global_mean_;
}