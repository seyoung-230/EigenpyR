# user: 0,1,2,3 / item: 0,1,2,3,4 (0-indexed - matches the underlying C++ class)
ratings <- rbind(
  c(0, 0, 5.0),
  c(0, 1, 4.0),
  c(0, 3, 1.0),
  c(1, 0, 4.0),
  c(1, 1, 5.0),
  c(1, 2, 1.0),
  c(2, 0, 1.0),
  c(2, 2, 5.0),
  c(2, 4, 4.0),
  c(3, 1, 1.0),
  c(3, 3, 5.0),
  c(3, 4, 4.0)
)

n_users <- 4
n_items <- 5

# construct and fit
mf <- new(MatrixFactorizationSGD,
  n_users,
  n_items,
  3,      # n_factors
  0.01,   # learning rate
  0.02,   # regularization
  100,    # n_epochs
  123     # seed
)

mf$fit(ratings, TRUE)

cat("\nPredicted rating matrix:\n")
print(mf$full_prediction())

cat("\nPrediction for user 0, item 2:", mf$predict(0, 2), "\n")
cat("Prediction for user 2, item 1:", mf$predict(2, 1), "\n")
