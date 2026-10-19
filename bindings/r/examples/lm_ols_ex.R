set.seed(123)
n = 100
p = 2
X = matrix(rnorm(n * p), n, p)
beta_true = c(2, -1)
intercept = 1
y = drop(1 + X %*% beta_true + 0.1 * rnorm(n))

# construct and fit
lm_ols = new(LmOLS)
lm_ols$fit(X, y)

print(lm_ols$intercept())
print(lm_ols$coef())

# prediction
X_new = matrix(c(1, 2, 3, 4, 5, 0, 0, 0, 0, 0), nrow = 5, ncol = 2)
pred = lm_ols$predict(X_new)
print(pred)

plot(X_new[, 1], 1 + 2 * X_new[, 1])
lines(X_new[, 1], pred, col = "blue")
