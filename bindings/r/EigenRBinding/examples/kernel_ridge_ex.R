set.seed(1)

ftrue <- function(x) sin(2 * pi * x) + 0.5 * cos(4 * pi * x)

n <- 150
X <- matrix(0, n, 1)
y <- numeric(n)
for (i in 1:n) {
  x <- runif(1, -1, 1)
  X[i, 1] <- x
  y[i] <- ftrue(x) + rnorm(1, 0, 0.1)
}

lambda_ <- 1e-3
sigma <- 0.2

# construct and fit
kr <- new(KernelRidge, lambda_, sigma)
kr$fit(X, y)

fitted <- kr$predict(X)
train_mse <- mean((fitted - y)^2)

cat("Kernel Ridge Regression example\n")
cat("n =", n, "\n")
cat("lambda =", lambda_, "\n")
cat("sigma =", sigma, "\n")
cat("y_mean =", kr$y_mean(), "\n")
cat("Train MSE =", train_mse, "\n")

# prediction on a grid
m <- 10
X_grid <- matrix(seq(-1, 1, length.out = m), m, 1)
pred <- kr$predict(X_grid)

cat("\nPredictions on grid:\n")
for (i in 1:m) {
  cat(sprintf("x = %.4f, yhat = %.4f\n", X_grid[i, 1], pred[i]))
}

# plot: noisy data, true function, fitted curve
X_plot <- matrix(seq(-1, 1, length.out = 200), 200, 1)
y_true_plot <- ftrue(X_plot[, 1])
y_pred_plot <- kr$predict(X_plot)

plot(X[, 1], y, pch = 16, col = "grey60",
     xlab = "x", ylab = "y", main = "Kernel Ridge Regression")
lines(X_plot[, 1], y_true_plot, col = "black", lwd = 2, lty = 2)
lines(X_plot[, 1], y_pred_plot, col = "red", lwd = 2)
legend("topright",
       legend = c("noisy data", "true function", "KRR fit"),
       col = c("grey60", "black", "red"),
       pch = c(16, NA, NA),
       lty = c(NA, 2, 1),
       lwd = c(NA, 2, 2))
