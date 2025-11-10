#include <gtest/gtest.h>
#include "Matrix.h"

TEST(MatrixTest, Addition) {
    Matrix A(2,2), B(2,2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    Matrix C = A + B;
    EXPECT_EQ(C(0,0), 6);
    EXPECT_EQ(C(1,1), 12);
}

TEST(MatrixTest, MultiplyScalar) {
    Matrix A(2,2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix C = A * 2.0;
    EXPECT_EQ(C(0,0), 2);
    EXPECT_EQ(C(1,1), 8);
}

TEST(MatrixTest, DotProductBasic) {
    Matrix A(1,3), B(3,1);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    B(0,0)=4; B(1,0)=5; B(2,0)=6;
    Matrix C = A * B;
    EXPECT_EQ(C.getRows(), 1);
    EXPECT_EQ(C.getCols(), 1);
    EXPECT_EQ(C(0,0), 1*4 + 2*5 + 3*6);
}

