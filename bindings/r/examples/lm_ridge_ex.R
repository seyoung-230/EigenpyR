set.seed(123)
n = 100
p = 10
X = matrix(rnorm(n * p), n, p)
beta_true = c(2, -1, rep(0, p - 2))
y = drop(X %*% beta_true + rnorm(n))

# construct and fit
ridge = new(LmRidge, 0.5)
ridge$fit(X, y)
 
print(ridge$coef())

# prediction
X_new = X[1:5, ]
pred = ridge$predict(X_new)
print(pred)

