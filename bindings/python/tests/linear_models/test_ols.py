import numpy as np
import matplotlib.pyplot as plt
import eigenpyr

# reproducibility
np.random.seed(123)

# data generation
n = 100
p = 2
X = np.random.randn(n, p)

beta_true = np.array([2.0, -1.0])
intercept = 1.0
y = intercept + X @ beta_true + 0.1 * np.random.randn(n)

# construct and fit
lm_ols = eigenpyr.LmOLS()
lm_ols.fit(X, y)

print("intercept =", lm_ols.intercept)
print("coef =", lm_ols.coef)

# prediction
X_new = np.array([
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0]
], dtype=float)

pred = lm_ols.predict(X_new)
print("pred =", pred)

# plot
plt.plot(X_new[:, 0], 1 + 2 * X_new[:, 0], label="true line")
plt.plot(X_new[:, 0], pred, label="predicted line")
plt.xlabel("X_new[:, 0]")
plt.ylabel("y")
plt.legend()
plt.show()


