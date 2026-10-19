#' Kernel Ridge Regression (Gaussian/RBF Kernel)
#'
#' Solves ridge regression in the feature space induced by
#' \code{k(a, b) = exp(-||a - b||^2 / (2 * sigma^2))}, giving the dual
#' solution \code{alpha = (K + lambda * I)^-1 (y - mean(y))}. The training
#' inputs are retained on the fitted object since prediction needs them.
#'
#' @param lambda Non-negative ridge penalty applied in the dual problem.
#' @param sigma Gaussian kernel bandwidth; larger values give smoother fits.
#'
#' @return An \code{Rcpp_KernelRidge} reference object with methods:
#' \describe{
#'   \item{\code{$fit(X, y)}}{Fit the model by solving the regularised dual system.}
#'   \item{\code{$predict(Xnew)}}{Predict responses for new observations.}
#'   \item{\code{$alpha()}}{Dual coefficients.}
#'   \item{\code{$x_train()}}{Training design matrix retained for prediction.}
#'   \item{\code{$lambda_()}}{The ridge penalty this model was constructed with.}
#'   \item{\code{$sigma()}}{The Gaussian kernel bandwidth this model was constructed with.}
#'   \item{\code{$y_mean()}}{Mean of the training responses, added back when predicting.}
#' }
#'
#' @examples
#' \dontrun{
#' m <- new(KernelRidge, 1e-3, 0.2)
#' m$fit(X, y)
#' m$predict(X_new)
#' }
#'
#' @name KernelRidge
#' @export KernelRidge
NULL
