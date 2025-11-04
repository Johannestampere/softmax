#include "SoftmaxRegressor.h"
#include <iostream>

SoftmaxRegressor::SoftmaxRegressor(size_t n_features, size_t n_classes, double learning_rate):
    theta{n_features, n_classes}, b{1, n_classes}, eta{learning_rate} {}

Matrix SoftmaxRegressor::predict_proba(const Matrix& X) const {
    // Z = X * theta + b
    // compute raw scores, each row = raw class scores for a sample
    Matrix Z = X * theta;

    // add bias
    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            Z(i, j) += b(0, j);
        }
    }

    // subtract row's max from every val in row to avoid overflow (probs stay the same)
    Matrix rowMax = Z.maxRowwise();

    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            Z(i, j) -= rowMax(i, 0);
        }
    }

    // exp to pass into softmax
    Matrix expZ = Z.exp();

    // softmax
    Matrix rowSum = expZ.sumRows();
    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            expZ(i, j) /= rowSum(i, 0);
        }
    }

    // returns a matrix where every row is one sample and each col in a row is the probability of that sample being in that class
    return expZ;
}

Matrix SoftmaxRegressor::predict(const Matrix& X) const {
    Matrix probabilities = predict_proba(X);
    Matrix res{probabilities.getRows(), 1};

    for (size_t i = 0; i < res.getRows(); ++i) {
        double maxVal = probabilities(i, 0);
        size_t maxIndex = 0;

        for (size_t j = 1; j < probabilities.getCols(); ++j) {
            if (probabilities(i, j) > maxVal) {
                maxVal = probabilities(i, j);
                maxIndex = j;
            }
        }

        res(i, 0) = static_cast<double>(maxIndex);
    }

    return res;
}

void SoftmaxRegressor::fit(const Matrix& X, const Matrix& y_onehot, int epochs) {

    size_t m = X.getRows(); // N(samples)

    for (int e = 0; e < epochs; ++e) {
        Matrix P = predict_proba(X);

        Matrix diff = P - y_onehot; // pred error per sample

        Matrix gradientTheta = (X.transpose() * diff) / static_cast<double>(m);
        Matrix gradientB = diff.sumCols() / static_cast<double>(m);

        // GD update
        theta = theta - gradientTheta * eta;
        b = b - gradientB * eta;

        // print loss
        if (e % 10 == 0) {
            double loss = compute_loss(X, y_onehot);
            cout << "Epoch " << e << " | Loss: " << loss << endl;
        }
    }
}

double SoftmaxRegressor::compute_loss(const Matrix& X, const Matrix& y_onehot) const {
    size_t m = X.getRows();

    Matrix P = predict_proba(X);

    // compute element-wise log of probs
    Matrix logP = P.log(1e-12);

    // elementwise multiply y_onehot * logP
    double total = 0.0;
    for (size_t i = 0; i < y_onehot.getRows(); ++i) {
        for (size_t j = 0; j < y_onehot.getCols(); ++j) {
            total += y_onehot(i, j) * logP(i, j);
        }
    }

    return -total / static_cast<double>(m);
}