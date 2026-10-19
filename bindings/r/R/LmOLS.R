#' Ordinary Least Squares Regression
#'
#' Fits \code{y = X \%*\% beta + intercept} by minimizing the residual sum of
#' squares. The intercept is derived from the column means, so \code{X}
#' should not include an intercept column of its own.
#'
#' @return An \code{Rcpp_LmOLS} reference object with methods:
#' \describe{
#'   \item{\code{$fit(X, y)}}{Estimate the regression coefficients from training data.}
#'   \item{\code{$predict(Xnew)}}{Predict responses for new observations.}
#'   \item{\code{$coef()}}{Estimated slope coefficients.}
#'   \item{\code{$intercept()}}{Estimated intercept term.}
#' }
#'
#' @examples
#' \dontrun{
#' m <- new(LmOLS)
#' m$fit(X, y)
#' m$predict(X_new)
#' }
#'
#' @name LmOLS
#' @export LmOLS
NULL
