#include <RcppEigen.h>
#include <utils.h>
#include <lm_ols.h>
#include <lm_ridge.h>
#include <lm_lasso.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]

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
  

  
}
