#include "Matrix.h"
#include <cmath>
#include <iomanip>
#include <stdexcept>

Matrix::Matrix(size_t r, size_t c): rows{r}, cols{c}, data(r * c, 0.0) {}

double& Matrix::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw invalid_argument("Matrix dimensions must match for addition.");

    Matrix result{rows, cols};

    for (size_t i = 0; i < (rows * cols); ++i) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw invalid_argument("Matrix dimensions must match for addition.");
    
    Matrix result{rows, cols};

    for (size_t i = 0; i < (rows * cols); ++i) {
        result.data[i] = data[i] - other.data[i];
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result{rows, cols};

    for (size_t i = 0; i < (rows * cols); ++i) {
        result.data[i] = data[i] * scalar;
    }

    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result{rows, cols};

    for (size_t i = 0; i < (rows * cols); ++i) {
        result.data[i] = data[i] / scalar;
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) throw invalid_argument("Matrix multiplication dimension mismatch");

    Matrix result{rows, other.cols};

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result{cols, rows};

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

Matrix Matrix::exp() const {
    Matrix result{rows, cols};

    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = std::exp(data[i]);
    }

    return result;
}

Matrix Matrix::log(double eps) const {
    Matrix result{rows, cols};

    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = std::log(data[i] + eps);
    }

    return result;
}

Matrix Matrix::sumRows() const {
    Matrix result{rows, 1};

    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
            sum += (*this)(i, j);
        result(i, 0) = sum;
    }

    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result{1, cols};

    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i)
            sum += (*this)(i, j);
        result(0, j) = sum;
    }

    return result;
}

Matrix Matrix::maxRowwise() const {
    Matrix result{rows, 1};

    for (size_t i = 0; i < rows; ++i) {
        double maxVal = (*this)(i, 0);
        for (size_t j = 1; j < cols; ++j)
            if ((*this)(i, j) > maxVal)
                maxVal = (*this)(i, j);
        result(i, 0) = maxVal;
    }

    return result;
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
