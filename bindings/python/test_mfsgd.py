import sys
sys.path.append(r"C:\Users\User\Desktop\230\eigenPyR-git\EigenPyR\build\Debug")

import eigenpyr

# user: 0,1,2,3
# item: 0,1,2,3,4
ratings = [
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating(),
    eigenpyr.Rating()
]

ratings[0].user = 0
ratings[0].item = 0
ratings[0].value = 5.0

ratings[1].user = 0
ratings[1].item = 1
ratings[1].value = 4.0

ratings[2].user = 0
ratings[2].item = 3
ratings[2].value = 1.0

ratings[3].user = 1
ratings[3].item = 0
ratings[3].value = 4.0

ratings[4].user = 1
ratings[4].item = 1
ratings[4].value = 5.0

ratings[5].user = 1
ratings[5].item = 2
ratings[5].value = 1.0

ratings[6].user = 2
ratings[6].item = 0
ratings[6].value = 1.0

ratings[7].user = 2
ratings[7].item = 2
ratings[7].value = 5.0

ratings[8].user = 2
ratings[8].item = 4
ratings[8].value = 4.0

ratings[9].user = 3
ratings[9].item = 1
ratings[9].value = 1.0

ratings[10].user = 3
ratings[10].item = 3
ratings[10].value = 5.0

ratings[11].user = 3
ratings[11].item = 4
ratings[11].value = 4.0

n_users = 4
n_items = 5

model = eigenpyr.MatrixFactorizationSGD(
    n_users,
    n_items,
    3,      # n_factors
    0.01,   # learning rate
    0.02,   # regularization
    100,    # epochs
    123     # seed
)

model.fit(ratings, True)

print("\nPredicted rating matrix:")
print(model.full_prediction())
print()

print("Prediction for user 0, item 2:", model.predict(0, 2))
print("Prediction for user 2, item 1:", model.predict(2, 1))