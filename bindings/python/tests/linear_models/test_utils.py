import numpy as np
import eigenpyr

# matrix example
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

X_std, col_means, col_sds = eigenpyr.standardize_matrix(X)

print(col_means)
print()

print(col_sds)
print()

print(X)
print()

print(X_std)
print()

# vector example
y = np.array([1.0, 2.0, 3.0])

y_std, mean_y, sd_y = eigenpyr.standardize_vector(y)

print(mean_y)
print()

print(sd_y)
print()

print(y)
print()

print(y_std)
print()