#' Matrix Factorization Trained by Stochastic Gradient Descent
#'
#' Approximates a sparse ratings matrix as a global mean plus per-user and
#' per-item biases plus the inner product of low-rank user/item factor
#' matrices, trained with L2-regularized SGD over the observed ratings.
#'
#' @param n_users Number of distinct users; user indices must be below this.
#' @param n_items Number of distinct items; item indices must be below this.
#' @param n_factors Rank of the latent factor matrices.
#' @param lr SGD learning rate.
#' @param reg L2 regularisation strength applied to factors and biases.
#' @param n_epochs Number of passes over the training ratings.
#' @param seed Seed for the random number generator used to initialise factors.
#'
#' @return An \code{Rcpp_MatrixFactorizationSGD} reference object with methods:
#' \describe{
#'   \item{\code{$fit(ratings, verbose)}}{Train on an n x 3 (user, item, value) ratings matrix.}
#'   \item{\code{$predict(user, item)}}{Predict the rating a single user would give a single item.}
#'   \item{\code{$full_prediction()}}{Reconstruct the dense prediction matrix for every user-item pair.}
#'   \item{\code{$user_factors()}}{Learned user factor matrix.}
#'   \item{\code{$item_factors()}}{Learned item factor matrix.}
#'   \item{\code{$user_bias()}}{Learned per-user bias terms.}
#'   \item{\code{$item_bias()}}{Learned per-item bias terms.}
#'   \item{\code{$global_mean()}}{Mean of all training ratings, used as the prediction baseline.}
#' }
#'
#' @examples
#' \dontrun{
#' m <- new(MatrixFactorizationSGD, n_users, n_items, 3, 0.01, 0.02, 100, 123)
#' m$fit(ratings, TRUE)
#' m$predict(0, 2)
#' }
#'
#' @name MatrixFactorizationSGD
#' @export MatrixFactorizationSGD
NULL
