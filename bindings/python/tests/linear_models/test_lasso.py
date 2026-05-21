import sys
sys.path.append(r"C:\Users\User\Desktop\230\eigenPyR-git\EigenPyR\build\Debug")

import numpy as np
import matplotlib.pyplot as plt
import eigenpyr

# =========================
# LmOLS check
# =========================
np.random.seed(123)

n = 100
p = 2
X = np.random.randn(n, p)

beta_true = np.array([2.0, -1.0])
intercept = 1.0
y = intercept + X @ beta_true + 0.1 * np.random.randn(n)

lm_ols = eigenpyr.LmOLS()
lm_ols.fit(X, y)

print("=== LmOLS ===")
print("intercept =", lm_ols.intercept)
print("coef =", lm_ols.coef)

X_new = np.array([
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0]
], dtype=float)

pred = lm_ols.predict(X_new)
print("pred =", pred)

plt.plot(X_new[:, 0], 1 + 2 * X_new[:, 0], label="true line")
plt.plot(X_new[:, 0], pred, label="predicted line")
plt.xlabel("X_new[:, 0]")
plt.ylabel("y")
plt.legend()
plt.show()


# =========================
# LmLasso check
# =========================
print("\n=== LmLasso ===")

n = 100
p = 5

true_coef = np.array([1, 2, -1, 0, 0, 0], dtype=float)   # intercept 포함

rng = np.random.default_rng(1)
X = rng.uniform(-1.0, 1.0, size=(n, p))

y = true_coef[0] + 0.1 * rng.normal(size=n)
for j in range(p):
    y += X[:, j] * true_coef[j + 1]

lm_lasso = eigenpyr.LmLasso(0.03)
lm_lasso.fit(X, y)

print("true coefficients =", true_coef)
print("estimated coefficients =", np.concatenate(([lm_lasso.intercept], lm_lasso.coef)))

X_new = np.array([
    [1, 1, 2, -1, 0],
    [3, 1, 2,  1, 3],
    [4, 1, 1,  3, 1]
], dtype=float)

print("new data =")
print(X_new)

pred_lasso = lm_lasso.predict(X_new)
print("predicted_y =", pred_lasso)