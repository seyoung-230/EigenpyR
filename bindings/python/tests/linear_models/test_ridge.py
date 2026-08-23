import numpy as np
import eigenpyr

np.random.seed(0)

n = 100
p = 5

true_coef = np.array([1, 2, -1, 0, 0, 0], dtype=float)

X = np.random.uniform(-1.0, 1.0, size=(n, p))

y = np.zeros(n)
for i in range(n):
    y[i] = true_coef[0] + 0.1 * np.random.normal()
    for j in range(p):
        y[i] += X[i, j] * true_coef[j + 1]

lm_ridge = eigenpyr.LmRidge(0.03)
lm_ridge.fit(X, y)

print("true coefficients:", true_coef)
print()
print("estimated coefficients:", lm_ridge.intercept, lm_ridge.coef)
print()

X_new = np.array([
    [1, 1, 2, -1, 0],
    [3, 1, 2,  1, 3],
    [4, 1, 1,  3, 1]
], dtype=float)

print("new data:")
print(X_new)
print("predicted_y:", lm_ridge.predict(X_new))