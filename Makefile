CXX = g++
CXXFLAGS = -std=c++20 -Wall -O3 -Iheaders
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# enable AVX2/FMA on x86_64, other architectures will compile scalar fallbacks
ifeq ($(UNAME_M),x86_64)
CXXFLAGS += -mavx2 -mfma
endif

SRC_DIR = implementations
OBJ_DIR = build
BIN = softmax
MAIN = main.cpp

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

GTEST_DIR = external/googletest
GTEST_LIB = $(GTEST_DIR)/build/lib
GTEST_INC = $(GTEST_DIR)/googletest/include

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(MAIN)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(OBJ_DIR)/test_matrix.o $(OBJ_DIR)/test_softmax.o $(OBJS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_INC) $^ \
    $(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a -pthread \
    -o run_tests

$(OBJ_DIR)/test_matrix.o: tests/test_matrix.cpp
	$(CXX) $(CXXFLAGS) -I$(GTEST_INC) -c $< -o $@

$(OBJ_DIR)/test_softmax.o: tests/test_softmax.cpp
	$(CXX) $(CXXFLAGS) -I$(GTEST_INC) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)