#include "Matrix.h"
#include "SoftmaxRegressor.h"
#include <iostream>

using namespace std;

int main() {
    Matrix X(4, 2);
    X(0,0) = 1.0; X(0,1) = 2.0;
    X(1,0) = 2.0; X(1,1) = 1.0;
    X(2,0) = 3.0; X(2,1) = 4.0;
    X(3,0) = 4.0; X(3,1) = 3.0;

    Matrix y(4, 3);
    y(0,0) = 1; y(0,1) = 0; y(0,2) = 0; // low
    y(1,0) = 1; y(1,1) = 0; y(1,2) = 0; // low
    y(2,0) = 0; y(2,1) = 1; y(2,2) = 0; // medium
    y(3,0) = 0; y(3,1) = 0; y(3,2) = 1; // high

    SoftmaxRegressor model(2, 3, 0.1);

    model.fit(X, y, 1000);

    Matrix X_test(2, 2);
    X_test(0,0) = 2.5; X_test(0,1) = 3.0; // new student A
    X_test(1,0) = 0.5; X_test(1,1) = 1.0; // new student B

    Matrix probs = model.predict_proba(X_test);
    cout << "Predicted probabilities:\n";
    probs.print();

    Matrix preds = model.predict(X_test);
    cout << "Predicted classes:\n";
    preds.print();

    return 0;
}