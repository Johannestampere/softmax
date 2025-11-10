# Softmax Regression from Scratch (C++)

A complete **Softmax (Multinomial Logistic) Regression** model implemented entirely in modern C++, without external ML or linear algebra libraries.

The implementation includes both standard scalar numerical routines and optional **SIMD-accelerated CPU paths** using **AVX2 and FMA intrinsics** for improved performance on supported architectures.

This project reproduces the mathematical foundations behind scikit-learn’s **LogisticRegression(multi_class='multinomial')**, including:
- Matrix operations (add, subtract, multiply, transpose, exp/log)
- Softmax normalization and cross-entropy loss
- Gradient descent optimization
- CPU-efficient vectorized execution where available

---

## Overview

Softmax Regression generalizes Logistic Regression to handle multiple classes.  
It computes a probability distribution over all possible classes using the **softmax** function:

$$
P(y = k \mid x) = \frac{e^{x \cdot \theta_k + b_k}}{\sum_j e^{x \cdot \theta_j + b_j}}
$$

The model is trained by minimizing the **cross-entropy loss** using **batch gradient descent**.

$$
J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} 
y_i^{(k)} \ \log \ p_i^{(k)}
$$

---

## Features

Fully custom `Matrix` class with:
- Dynamic allocation and bounds-safe element access  
- Matrix arithmetic (`+`, `-`, `*`, `/`)  
- Transpose, elementwise `exp()` / `log()`  
- Row/column reductions (`sumRows`, `sumCols`, `maxRowwise`)  
- AVX2-accelerated elementwise operations (addition, subtraction, scalar multiply/divide)
- AVX2/FMA-optimized matrix multiplication, including a transposed right-hand operand for improved cache locality
- Automatic scalar fallback on CPUs without AVX2 support

`SoftmaxRegressor` implementation:
- Vectorized forward pass: `Z = XΘ + b`
- Numerically stable softmax normalization
- Uses SIMD-based arithmetic when available  
- Cross-entropy loss calculation  
- Gradient descent parameter updates  
- Predicts both probabilities and class labels  

---

## Build Instructions

```bash
# Clone the repo
git clone https://github.com/johannestampere/softmax.git
cd softmax

# Build
make

# Run
./softmax
