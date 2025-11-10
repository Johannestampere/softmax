#include <gtest/gtest.h>
#include "SoftmaxRegressor.h"
#include "Matrix.h"

TEST(SoftmaxTest, PredictSimpleTwoClass) {
    SoftmaxRegressor clf(2, 2, 0.1);

    Matrix X(4,2);
    X(0,0)=0; X(0,1)=0;
    X(1,0)=0; X(1,1)=1;
    X(2,0)=1; X(2,1)=0;
    X(3,0)=1; X(3,1)=1;

    Matrix y(4,2);
    y(0,0)=1; y(0,1)=0;
    y(1,0)=0; y(1,1)=1;
    y(2,0)=0; y(2,1)=1;
    y(3,0)=1; y(3,1)=0;
    clf.fit(X, y, 200, 2);

    Matrix pred = clf.predict(X);
    
    EXPECT_EQ(pred(0,0), 0);
    EXPECT_EQ(pred(1,0), 1);
    EXPECT_EQ(pred(2,0), 1);
    EXPECT_EQ(pred(3,0), 0);
}

static double numerical_grad_theta(SoftmaxRegressor& clf, const Matrix& X, const Matrix& y, size_t r, size_t c) {
    double eps = 1e-5;
    Matrix& theta = clf.theta_ref();
    double old = theta(r,c);
    theta(r,c) = old + eps;
    double loss1 = clf.compute_loss(X, y);
    theta(r,c) = old - eps;
    double loss2 = clf.compute_loss(X, y);
    theta(r,c) = old;
    return (loss1 - loss2) / (2 * eps);
}

TEST(SoftmaxTest, GradientCorrectness) {
    SoftmaxRegressor clf(3, 2, 0.1);
    Matrix X(4, 3);
    Matrix y(4, 2);

    for (size_t i = 0;i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
            X(i, j) = 0.1 * (i + j + 1);
        }
            
    }
        
    y(0,0)=1; y(1,1)=1; y(2,1)=1; y(3,0)=1;

    Matrix P = clf.predict_proba(X);
    Matrix diff = P - y;
    Matrix grad_theta = (X.transpose() * diff) / 4.0;
    
    double num = numerical_grad_theta(clf, X, y, 0, 0);
    double ana = grad_theta(0,0);

    EXPECT_NEAR(num, ana, 1e-4);
}

