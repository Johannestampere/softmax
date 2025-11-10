# Softmax Regression from Scratch (C++)

A complete **Softmax (Multinomial Logistic) Regression** model built entirely in modern C++, using a fully custom matrix library and no external ML or linear algebra dependencies.

The implementation supports both standard scalar routines and **SIMD-accelerated CPU execution** using **AVX2** and **FMA** intrinsics when available.

---

## Overview

Softmax Regression generalizes Logistic Regression to multi-class problems.  
Given an input \( x \), class probabilities are computed using the softmax function:

$$
P(y = k \mid x) =
\frac{e^{x \cdot \theta_k + b_k}}
{\sum_j e^{x \cdot \theta_j + b_j}}
$$

Training minimizes the cross-entropy loss:

$$
J(\Theta) = -\frac{1}{m}
\sum_{i=1}^{m} \sum_{k=1}^{K}
y_i^{(k)} \log \, p_i^{(k)}
$$

The model uses **mini-batch gradient descent**, with dataset shuffling and per-batch updates for faster and more stable convergence.

---

## Features

### Matrix Library
- Dynamic allocation with row-major storage  
- Bounds-safe element access  
- Arithmetic operators (`+`, `-`, `*`, `/`)  
- Transpose, elementwise `exp()` and `log()`  
- Reductions: `sumRows`, `sumCols`, `maxRowwise`  

### Performance Features
- **AVX2-accelerated elementwise operations**  
- **AVX2/FMA-optimized matrix multiplication** with transposed RHS for improved cache locality  
- Scalar fallback path on CPUs without AVX2 support  

### SoftmaxRegressor
- Forward pass: `Z = XΘ + b`  
- Numerically stable softmax  
- **Mini-batch gradient descent** with shuffled batches  
- Cross-entropy loss computation  
- Probability and class label prediction  

### Unit Testing
- GoogleTest-based tests for matrix operations  
- Prediction correctness tests  
- Numerical gradient checking  
- Test target integrated into the Makefile (`make test`)

---

## Build Instructions

```bash
git clone https://github.com/johannestampere/softmax.git
cd softmax
make
./softmax
```