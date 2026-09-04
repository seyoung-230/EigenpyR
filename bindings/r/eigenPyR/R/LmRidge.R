#' Ridge (L2-Penalized) Linear Regression
#'
#' Minimizes the residual sum of squares plus \code{lambda} times the
#' squared L2 norm of the slope coefficients. The intercept is not
#' penalized.
#'
#' @param lambda Non-negative L2 penalty strength; 0 reduces to OLS.
#'
#' @return An \code{Rcpp_LmRidge} reference object with methods:
#' \describe{
#'   \item{\code{$fit(X, y)}}{Estimate the regression coefficients from training data.}
#'   \item{\code{$predict(Xnew)}}{Predict responses for new observations.}
#'   \item{\code{$coef()}}{Estimated slope coefficients.}
#'   \item{\code{$intercept()}}{Estimated intercept term.}
#' }
#'
#' @examples
#' \dontrun{
#' m <- new(LmRidge, 0.1)
#' m$fit(X, y)
#' m$predict(X_new)
#' }
#'
#' @name LmRidge
#' @export LmRidge
NULL
