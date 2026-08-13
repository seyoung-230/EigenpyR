import sys
sys.path.append(r"C:\Users\User\Desktop\230\eigenPyR\260809\EigenpyR\build\Debug")

import numpy as np
import eigenpyr

PI = np.pi

def ftrue(x):
    return np.sin(2.0 * PI * x) + 0.5 * np.cos(4.0 * PI * x)

np.random.seed(1)

n = 150
X = np.zeros((n, 1))
y = np.zeros(n)

for i in range(n):
    x = np.random.uniform(-1.0, 1.0)
    X[i, 0] = x
    y[i] = ftrue(x) + np.random.normal(0.0, 0.1)

lambda_ = 1e-3
sigma = 0.2

model = eigenpyr.KernelRidge(lambda_, sigma)
model.fit(X, y)

fitted = model.predict(X)
train_mse = np.mean((fitted - y) ** 2)

print("Kernel Ridge Regression example")
print("n =", n)
print("lambda =", lambda_)
print("sigma =", sigma)
print("y_mean =", model.y_mean)
print("Train MSE =", train_mse)

m = 10
X_grid = np.zeros((m, 1))
for i in range(m):
    X_grid[i, 0] = -1.0 + 2.0 * i / (m - 1.0)

pred = model.predict(X_grid)

print("\nPredictions on grid:")
for i in range(m):
    print(f"x = {X_grid[i, 0]}, yhat = {pred[i]}")