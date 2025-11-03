#include "SoftmaxRegressor.h"

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
    Matrix res{probabilites.getRows(), 1};

    // get max of 
    for (size_t i = 0; i < res.getRows(); ++i) {
        double maxVal = probabilities(i, 0);
        size_t maxIndex = 0;

        for (size_t j = 1; j < probs.numCols(); ++j) {
            if (probabilities(i, j) > maxVal) {
                maxVal = probabilities(i, j);
                maxIdx = j;
            }
        }

        res(i, 0) = static_cast<double>(maxIdx);
    }
}

