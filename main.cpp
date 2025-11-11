#include "Matrix.h"
#include "SoftmaxRegressor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
using namespace std;

const int PIXEL_RANGE = 255;

struct MnistData {
    Matrix X;
    vector<int> labels;
};

static size_t count_lines(const string& path) {
    ifstream in(path);
    size_t n = 0;
    string line;
    while (getline(in, line)) ++n;
    return n;
}

static MnistData load_mnist_csv(const string& path) {
    const size_t cols = 784;
    size_t rows = count_lines(path);

    ifstream in(path);
    if (!in) {
        throw runtime_error("Failed to open MNIST CSV: " + path);
    }

    // ignore header row
    string header;
    getline(in, header);

    Matrix X(rows, cols);
    vector<int> labels(rows);

    string line;
    size_t r = 0;

    while (getline(in, line)) {
        stringstream ss(line);
        string cell;

        if (!getline(ss, cell, ',')) throw runtime_error("Invalid MNIST row (missing label)");
        labels[r] = stoi(cell);

        for (size_t c = 0; c < cols; ++c) {
            if (!getline(ss, cell, ',')) {
                throw runtime_error("Invalid MNIST row (missing pixel)");
            }
            double v = stod(cell) / static_cast<double>(PIXEL_RANGE); // normalization
            X(r, c) = v;
        }

        ++r;
    }

    return { std::move(X), std::move(labels) };
}

static Matrix one_hot(const vector<int>& labels, size_t num_classes) {
    Matrix Y(labels.size(), num_classes);

    for (size_t i = 0; i < labels.size(); ++i) {
        Y(i, static_cast<size_t>(labels[i])) = 1.0;
    }

    return Y;
}

static double accuracy(const Matrix& preds, const vector<int>& labels) {
    size_t correct = 0;

    for (size_t i = 0; i < preds.getRows(); ++i) {
        int p = static_cast<int>(preds(i, 0));
        if (p == labels[i]) ++correct;
    }

    return static_cast<double>(correct) / static_cast<double>(labels.size());
}

int main() {
    const string train_csv = "data/mnist_train.csv";
    const string test_csv  = "data/mnist_test.csv";

    cout << "Loading MNIST training CSV..." << endl;
    MnistData train = load_mnist_csv(train_csv);

    cout << "Loading MNIST test CSV..." << endl;
    MnistData test = load_mnist_csv(test_csv);

    const size_t n_features = 784;
    const size_t n_classes  = 10;

    cout << "One-hot encoding labels..." << endl;
    Matrix y_train = one_hot(train.labels, n_classes);

    SoftmaxRegressor model(n_features, n_classes, 0.1);

    cout << "Training model..." << endl;
    model.fit(train.X, y_train, 100, 256);

    cout << "Evaluating on test set..." << endl;
    Matrix preds = model.predict(test.X);

    double acc = accuracy(preds, test.labels);
    cout << "Test accuracy: " << acc * 100.0 << "%" << endl;

    return 0;
}