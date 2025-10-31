#pragma once
#include "Matrix.h"

class SoftmaxRegressor {
private:
    Matrix Theta; // parameter matrix (N(features) x N(classes))
    // every columns is a theta vector for one class

    Matrix b; // bias vector (1 x N(classes))
    double eta; // learning rate

public:

    SoftmaxRegressor(size_t n_features, size_t n_classes, double learning_rate=0.01);

    // returns predicted class probabilites for every sample in X
    //      1) computes Z = X * theta + b (bias vector)
    //      2) subtract each row's maximum value from every val in row
    //      3) exponentiate the scores
    //      4) normalize per row (softmax)
    Matrix predict_proba(const Matrix& X) const;

    // returns the predicted class labels for each input sample as a (n_samples x 1) matrix
    Matrix predict(const Matrix& X) const;

    // training
    // y_onehot has shape (n_samples x n_classes)
    // a 1 in each row at the true class position
    void fit(const Matrix& X, const Matrix& y_onehot, int epochs=1000);

    // loss
    double compute_loss(const Matrix& X, const Matrix& y_onehot) const;
};
