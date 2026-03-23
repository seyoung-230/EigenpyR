#include <iostream>
#include <vector>
#include "mf_sgd.h"

using namespace std;

int main() {
    // user: 0,1,2,3
    // item: 0,1,2,3,4
    vector<Rating> ratings = {
        {0, 0, 5.0}, {0, 1, 4.0}, {0, 3, 1.0},
        {1, 0, 4.0}, {1, 1, 5.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 2, 5.0}, {2, 4, 4.0},
        {3, 1, 1.0}, {3, 3, 5.0}, {3, 4, 4.0}
    };

    int n_users = 4;
    int n_items = 5;

    MatrixFactorizationSGD model(
        n_users,
        n_items,
        3,      // n_factors
        0.01,   // learning rate
        0.02,   // regularization
        100,    // epochs
        123     // seed
    );

    model.fit(ratings, true);

    cout << "\nPredicted rating matrix:\n";
    cout << model.full_prediction() << "\n\n";

    cout << "Prediction for user 0, item 2: "
              << model.predict(0, 2) << endl;

    cout << "Prediction for user 2, item 1: "
              << model.predict(2, 1) << endl;

    return 0;
}