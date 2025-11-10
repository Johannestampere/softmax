#include "Matrix.h"
#include <cmath>
#include <iomanip>
#include <stdexcept>
#ifdef __AVX2__
#include <immintrin.h>
#endif

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

    size_t n = rows * cols;
#ifdef __AVX2__
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&data[i]);
        __m256d vb = _mm256_loadu_pd(&other.data[i]);
        __m256d vs = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result.data[i], vs);
    }
    for (; i < n; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
#endif

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) throw invalid_argument("Matrix dimensions must match for addition.");
    
    Matrix result{rows, cols};

    size_t n = rows * cols;
#ifdef __AVX2__
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&data[i]);
        __m256d vb = _mm256_loadu_pd(&other.data[i]);
        __m256d vd = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result.data[i], vd);
    }
    for (; i < n; ++i) {
        result.data[i] = data[i] - other.data[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result.data[i] = data[i] - other.data[i];
    }
#endif

    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result{rows, cols};

    size_t n = rows * cols;
#ifdef __AVX2__
    __m256d vs = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&data[i]);
        __m256d vm = _mm256_mul_pd(va, vs);
        _mm256_storeu_pd(&result.data[i], vm);
    }
    for (; i < n; ++i) {
        result.data[i] = data[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result.data[i] = data[i] * scalar;
    }
#endif

    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result{rows, cols};

    size_t n = rows * cols;
#ifdef __AVX2__
    __m256d vs = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&data[i]);
        __m256d vd = _mm256_div_pd(va, vs);
        _mm256_storeu_pd(&result.data[i], vd);
    }
    for (; i < n; ++i) {
        result.data[i] = data[i] / scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result.data[i] = data[i] / scalar;
    }
#endif

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) throw invalid_argument("Matrix multiplication dimension mismatch");

    Matrix result{rows, other.cols};

#ifdef __AVX2__
    // transpose other to improve cache locality for dot products
    Matrix Bt = other.transpose();

    for (size_t i = 0; i < rows; ++i) {
        const double* arow = &data[i * cols];

        for (size_t j = 0; j < other.cols; ++j) {
            const double* brow = &Bt.data[j * cols];
            __m256d acc = _mm256_setzero_pd();
            size_t k = 0;

            for (; k + 4 <= cols; k += 4) {
                __m256d va = _mm256_loadu_pd(arow + k);
                __m256d vb = _mm256_loadu_pd(brow + k);

                #ifdef __FMA__
                acc = _mm256_fmadd_pd(va, vb, acc);
                #else
                acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
                #endif
            }
            double sum = 0.0;

            alignas(32) double tmp[4];

            _mm256_storeu_pd(tmp, acc);

            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
            
            for (; k < cols; ++k) {
                sum += arow[k] * brow[k];
            }

            result(i, j) = sum;
        }
    }
#else
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;

            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }

            result(i, j) = sum;
        }
    }
#endif

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
