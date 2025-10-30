#pragma once
#include "Matrix.h"

class SoftmaxRegressor {
private:
    Matrix Theta; // parameter matrix (N(features) x N(classes))
    Matrix b; // bias vector (1 x N(classes))
    double eta; // learning rate
};