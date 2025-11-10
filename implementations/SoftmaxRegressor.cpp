#include "SoftmaxRegressor.h"
#include <iostream>
#if __AVX2__
#include <immintrin.h>
#endif
#include <random>
#include <numeric>
#include <algorithm>

using namespace std;

static vector<size_t> shuffled_indices(size_t n) {
    vector<size_t> idx(n);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), mt19937{random_device{}()});
    return idx;
}

static Matrix sliceRows(const Matrix& M, const vector<size_t>& idx, size_t start, size_t count) {
    size_t cols = M.getCols();
    Matrix out(count, cols);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = M(idx[start + i], j);
        }
    }
    return out;
}

SoftmaxRegressor::SoftmaxRegressor(size_t n_features, size_t n_classes, double learning_rate):
    theta{n_features, n_classes}, b{1, n_classes}, eta{learning_rate} {}

Matrix SoftmaxRegressor::predict_proba(const Matrix& X) const {
    // Z = X * theta + b
    // compute raw scores, each row = raw class scores for a sample
    Matrix Z = X * theta;

    // add bias
#ifdef __AVX2__
    {
        const size_t rows = Z.getRows();
        const size_t cols = Z.getCols();
        const double* bptr = &b(0, 0);

        for (size_t i = 0; i < rows; ++i) {
            double* zrow = &Z(i, 0);
            size_t j = 0;

            for (; j + 4 <= cols; j += 4) {
                __m256d vz = _mm256_loadu_pd(zrow + j);
                __m256d vbias = _mm256_loadu_pd(bptr + j);
                __m256d vsum = _mm256_add_pd(vz, vbias);
                _mm256_storeu_pd(zrow + j, vsum);
            }

            for (; j < cols; ++j) {
                zrow[j] += bptr[j];
            }
        }
    }
#else
    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            Z(i, j) += b(0, j);
        }
    }
#endif

    // subtract row's max from every val in row to avoid overflow (probs stay the same)
    Matrix rowMax = Z.maxRowwise();

    #ifdef __AVX2__
    {
        const size_t rows = Z.getRows();
        const size_t cols = Z.getCols();

        for (size_t i = 0; i < rows; ++i) {
            double* zrow = &Z(i, 0);
            __m256d vmax = _mm256_set1_pd(rowMax(i, 0));
            size_t j = 0;

            for (; j + 4 <= cols; j += 4) {
                __m256d vz = _mm256_loadu_pd(zrow + j);
                __m256d vdiff = _mm256_sub_pd(vz, vmax);
                _mm256_storeu_pd(zrow + j, vdiff);
            }

            for (; j < cols; ++j) {
                zrow[j] -= rowMax(i, 0);
            }
        }
    }
    #else
    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            Z(i, j) -= rowMax(i, 0);
        }
    }
    #endif

    // exp to pass into softmax
    Matrix expZ = Z.exp();

    // softmax
    Matrix rowSum = expZ.sumRows();
    #ifdef __AVX2__
    {
        const size_t rows = expZ.getRows();
        const size_t cols = expZ.getCols();

        for (size_t i = 0; i < rows; ++i) {
            double* zrow = &expZ(i, 0);
            __m256d vden = _mm256_set1_pd(rowSum(i, 0));
            size_t j = 0;

            for (; j + 4 <= cols; j += 4) {
                __m256d vz = _mm256_loadu_pd(zrow + j);
                __m256d vq = _mm256_div_pd(vz, vden);
                _mm256_storeu_pd(zrow + j, vq);
            }

            for (; j < cols; ++j) {
                zrow[j] /= rowSum(i, 0);
            }
        }
    }
    #else
    for (size_t i = 0; i < Z.getRows(); ++i) {
        for (size_t j = 0; j < Z.getCols(); ++j) {
            expZ(i, j) /= rowSum(i, 0);
        }
    }
    #endif

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

void SoftmaxRegressor::fit(const Matrix& X, const Matrix& y_onehot, int epochs, size_t batch_size) {
    size_t m = X.getRows();

    if (batch_size == 0) batch_size = 32;

    for (int e = 0; e < epochs; ++e) {
        auto idx = shuffled_indices(m);

        for (size_t start = 0; start < m; start += batch_size) {
            size_t count = min(batch_size, m - start);

            Matrix Xb = sliceRows(X, idx, start, count);
            Matrix yb = sliceRows(y_onehot, idx, start, count);
            Matrix P = predict_proba(Xb);
            Matrix diff = P - yb;
            Matrix gradientTheta = (Xb.transpose() * diff) / static_cast<double>(count);
            Matrix gradientB = diff.sumCols() / static_cast<double>(count);

            theta = theta - gradientTheta * eta;
            b = b - gradientB * eta;
        }
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
