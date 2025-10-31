#pragma once
#include <Matrix.h>
#include <cmath>
#include <iomanip>
using namespace std;

Matrix::Matrix(size_t r, size_t c): rows{r}, cols{c}, data{r * c, 0.0} {}

double& Matrix::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw invalid_argument("Matrix dimensions must match for addition.");

    Matrix new{rows, cols};

    for (int i = 0; i < (rows * cols); ++i) {
        new.data[i] = data[i] + other.data[i];
    }

    return new;
}

Matrix operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw invalid_argument("Matrix dimensions must match for addition.");
    
    Matrix new{rows, cols};

    for (int i = 0; i < (rows * cols); ++i) {
        new.data[i] = data[i] - other.data[i];
    }

    return new;
}

Matrix operator*(double scalar) const {
    
    Matrix new{rows, cols};

    for (int i = 0; i < (rows * cols); ++i) {
        new.data[i] = data[i] * scalar;
    }

    return new;
}

Matrix operator/(double scalar) const {
    
    Matrix new{rows, cols};

    for (int i = 0; i < (rows * cols); ++i) {
        new.data[i] = data[i] / scalar;
    }

    return new;
}

Matrix operator*(const Matrix& other) const {
    if (cols != other.rows) throw invalid_argument("Matrix multiplication dimension mismatch");

    Matrix new{rows, other.cols};

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            new(i, j) = sum;
        }
    }

    return new;
}

Matrix Matrix::transpose() const {
    Matrix new{cols, rows};

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            new(j, i) = (*this)(i, j);
        }
    }

    return new;
}

Matrix Matrix::exp() const {
    Matrix new{rows, cols};

    for (size_t i = 0; i < rows * cols; ++i) {
        new.data[i] = exp(data[i]);
    }

    return new;
}

Matrix log(double eps=1e-12) const {
    Matrix new{rows, cols};

    for (size_t i = 0; i < rows * cols; ++i) {
        new.data[i] = log(data[i] + eps);
    }

    return new;
}

Matrix Matrix::sumRows() const {
    Matrix new{rows, 1};

    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
            sum += (*this)(i, j);
        new(i, 0) = sum;
    }

    return new;
}

Matrix Matrix::sumCols() const {
    Matrix new{1, cols};

    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i)
            sum += (*this)(i, j);
        new(0, j) = sum;
    }

    return new;
}

Matrix Matrix::maxRowwise() const {
    Matrix new{rows, 1};

    for (size_t i = 0; i < rows; ++i) {
        double maxVal = (*this)(i, 0);
        for (size_t j = 1; j < cols; ++j)
            if ((*this)(i, j) > maxVal)
                maxVal = (*this)(i, j);
        new(i, 0) = maxVal;
    }

    return new;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            cout << setw(10) << setprecision(5) << (*this)(i, j) << " ";
        }
        cout << "\n";
    }
    cout << endl;
}
