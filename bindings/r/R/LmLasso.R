#' Lasso (L1-Penalized) Linear Regression
#'
#' Minimizes the residual sum of squares plus \code{lambda} times the L1
#' norm of the slope coefficients, driving some of them exactly to zero.
#' Solved by cyclic coordinate descent.
#'
#' @param lambda Non-negative L1 penalty strength; larger values give
#'   sparser fits.
#'
#' @return An \code{Rcpp_LmLasso} reference object with methods:
#' \describe{
#'   \item{\code{$fit(X, y, max_iter, eps)}}{Estimate the regression coefficients by coordinate descent.}
#'   \item{\code{$predict(Xnew)}}{Predict responses for new observations.}
#'   \item{\code{$coef()}}{Estimated slope coefficients; may contain exact zeros.}
#'   \item{\code{$intercept()}}{Estimated intercept term.}
#' }
#'
#' @examples
#' \dontrun{
#' m <- new(LmLasso, 0.1)
#' m$fit(X, y, 500, 1e-6)
#' m$predict(X_new)
#' }
#'
#' @name LmLasso
#' @export LmLasso
NULL
