#include <RcppEigen.h>
#include <utils.h>
#include <lm_ols.h>
#include <lm_ridge.h>
#include <lm_lasso.h>
#include <kernel_ridge.h>
#include <mf_sgd.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]

// MatrixFactorizationSGD::fit() takes std::vector<Rating>, which Rcpp Modules
// cannot marshal directly from R. Accept an n x 3 (user, item, value) matrix
// instead and convert it here.
static std::vector<Rating> ratings_from_matrix(const Rcpp::NumericMatrix& m) {
  std::vector<Rating> ratings;
  ratings.reserve(m.nrow());
  for (int i = 0; i < m.nrow(); ++i) {
    ratings.push_back(Rating{ (int)m(i, 0), (int)m(i, 1), m(i, 2) });
  }
  return ratings;
}

static void mf_fit(MatrixFactorizationSGD* obj, const Rcpp::NumericMatrix& ratings, bool verbose) {
  obj->fit(ratings_from_matrix(ratings), verbose);
}

RCPP_MODULE(EigenRBinding_MODULE) {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  
  Rcpp::class_<LmOLS>("LmOLS")
  //.constructor<double>("Create LmOLS")
    .constructor("Create LmOLS")
    .method("fit",     &LmOLS::fit,     "fit(X, y)")
    .method("predict", &LmOLS::predict, "predict(Xnew)")
    .method("coef",    &LmOLS::coef,    "coefficients")
    .method("intercept",     &LmOLS::intercept,     "intercept");

  Rcpp::class_<LmRidge>("LmRidge")
  .constructor<double>("Create LmRidge")
  .method("fit",     &LmRidge::fit,     "fit(X, y)")
  .method("predict", &LmRidge::predict, "predict(Xnew)")
  .method("coef",    &LmRidge::coef,    "coefficients")
  .method("intercept",     &LmRidge::intercept,     "intercept");

  Rcpp::class_<LmLasso>("LmLasso")
      .constructor<double>("Create LmLasso")
      .method("fit",     &LmLasso::fit,     "fit(X, y, max_iter, eps)")
      .method("predict", &LmLasso::predict, "predict(Xnew)")
      .method("coef",    &LmLasso::coef,    "coefficients")
      .method("intercept",     &LmLasso::intercept,     "intercept");

  Rcpp::class_<KernelRidge>("KernelRidge")
      .constructor<double, double>("Create KernelRidge(lambda, sigma)")
      .method("fit",      &KernelRidge::fit,      "fit(X, y)")
      .method("predict",  &KernelRidge::predict,  "predict(Xnew)")
      .method("alpha",    &KernelRidge::alpha,    "dual coefficients")
      .method("x_train",  &KernelRidge::x_train,  "training X")
      .method("lambda_",  &KernelRidge::lambda,   "lambda")
      .method("sigma",    &KernelRidge::sigma,    "kernel bandwidth")
      .method("y_mean",   &KernelRidge::y_mean,   "y mean");

  Rcpp::class_<MatrixFactorizationSGD>("MatrixFactorizationSGD")
      .constructor<int, int, int, double, double, int, unsigned int>(
          "Create MatrixFactorizationSGD(n_users, n_items, n_factors, lr, reg, n_epochs, seed)")
      .method("fit",             &mf_fit,                                  "fit(ratings_matrix[,user,item,value], verbose)")
      .method("predict",         &MatrixFactorizationSGD::predict,         "predict(user, item)")
      .method("full_prediction", &MatrixFactorizationSGD::full_prediction, "full prediction matrix")
      .method("user_factors",    &MatrixFactorizationSGD::user_factors,    "user factors")
      .method("item_factors",    &MatrixFactorizationSGD::item_factors,    "item factors")
      .method("user_bias",       &MatrixFactorizationSGD::user_bias,       "user bias")
      .method("item_bias",       &MatrixFactorizationSGD::item_bias,       "item bias")
      .method("global_mean",     &MatrixFactorizationSGD::global_mean,     "global mean");
}
