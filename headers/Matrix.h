#pragma once
#include <vector>
#include <iostream>
#include <utility>

using namespace std;

class Matrix {
private:
    size_t rows, cols;
    vector<double> data;
public:
    Matrix(size_t r, size_t c);

    // modify matrix entries, eg A(0, 1) = 7.0; double x = A(73, 5);
    double& operator()(size_t r, size_t c);
    const double& operator()(size_t r, size_t c) const;

    // getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // basic arithmetic
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // basic operations
    Matrix operator*(const Matrix& other) const; // dot product
    Matrix transpose() const;
    
    // elementwise functions
    Matrix exp() const; // for softmax normalization
    Matrix log(double eps=1e-12) const; // for cross entropy loss

    // reductions
    Matrix sumRows() const; // sum across rows
    Matrix sumCols() const; // sum across cols
    Matrix maxRowwise() const; // get max from every row

    void print() const;
};



